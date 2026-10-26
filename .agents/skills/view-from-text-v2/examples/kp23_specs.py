"""KP23 (decision trees) pages expressed as layout specs -- the acceptance sample of view-from-text-v2.
Each function returns (layout_name, spec). Numbers are computed here, never typed; strings are {"zh","en"} pairs.
Use:  from figkit import layouts;  f = layouts.build(*impurity_examples());  f.to_nodes(page, 60, y, "dark");  f.to_png(path)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import T


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


def impurity_decrease():
    """Freeform infographic: the split drawn as icons, the bar of what is left, the formula with the numbers, a definition card."""
    from figkit.parts import NFig as _N, outline_box, icon, icon_rows, chip, panel, takeaway as _tk, tint as _tint, solid as _solid, header_chip
    par, yes, no = (4, 4, 2), (3, 1, 0), (1, 3, 2)
    Ip, Iy, In = H(par), H(yes), H(no); s1, s2 = 0.4 * Iy, 0.6 * In; d = Ip - s1 - s2
    f = _N("lec05-kp23-dI", 1160, 452, T(tr(
        f"不純度下降量：父節點 (4,4,2) 共 10 點、I={Ip:.3f}，被問題分成「是」(3,1,0) 4 點 I={Iy:.3f} 與「否」(1,3,2) 6 點 I={In:.3f}；ΔI=I(t)−(4/10)I(tY)−(6/10)I(tN)=1.522−0.325−0.875={d:.3f}；教材印 0.315。",
        f"Impurity decrease: the parent (4,4,2) has 10 points and I={Ip:.3f}; the question splits it into yes (3,1,0) with 4 points, I={Iy:.3f} and no (1,3,2) with 6 points, I={In:.3f}; ΔI=I(t)−(4/10)I(tY)−(6/10)I(tN)=1.522−0.325−0.875={d:.3f}; the textbook prints 0.315.")))
    def row(x, y, kinds):
        k = 0
        for kind, n in kinds:
            for _ in range(n):
                icon(f, kind, x + 14 + k * 28, y); k += 1
    # parent
    outline_box(f, 150, 0, 316, 56, "TXT", 1.5); row(158, 28, [("circle", 4), ("square", 4), ("tri", 2)])
    f.text(308, 70, T(tr("父節點 t = (4,4,2)：10 點，I = 1.522", "parent t = (4,4,2): 10 points, I = 1.522")), "TXT", 16, True, w=330)
    # children
    f.line(308, 88, 214, 112, "MUT", 2.5); f.line(308, 88, 402, 112, "MUT", 2.5)
    f.text(236, 100, T(tr("是", "yes")), "MUT", 16, True); f.text(380, 100, T(tr("否", "no")), "MUT", 16, True)
    outline_box(f, 150, 112, 128, 48, "TXT", 1.5); row(158, 136, [("circle", 3), ("square", 1)])
    outline_box(f, 338, 112, 184, 48, "TXT", 1.5); row(346, 136, [("circle", 1), ("square", 3), ("tri", 2)])
    f.text(214, 178, T(tr("「是」(3,1,0)：4 點", "yes (3,1,0): 4 points")), "TXT", 16, True, w=200); f.text(214, 200, f"I = {Iy:.3f}", "TXT", 16, w=200)
    f.text(430, 178, T(tr("「否」(1,3,2)：6 點", "no (1,3,2): 6 points")), "TXT", 16, True, w=220); f.text(430, 200, f"I = {In:.3f}", "TXT", 16, w=200)
    # definition card
    header_chip(f, 640, 0, 520, 34, "blue", T(tr("不純度下降量", "Impurity decrease")), size=20)
    panel(f, 640, 38, 520, 168, "blue")
    f.text(656, 112, T(tr("問完一個問題，節點分成「是」「否」兩堆；\n分完以後不純度少了多少，就用它比較問題，\n選 $\\Delta I$ 最大的那個。", "After a question the node splits into a yes and a no pile;\nhow much impurity is gone after the split compares the\nquestions, and the largest $\\Delta I$ wins.")), "TXT", 18, False, "left", w=490, math=True, name=T(tr("term:不純度下降量", "term:Impurity decrease")))
    f.text(656, 180, T(tr("$N_{tY}=4$、$N_{tN}=6$：「是」「否」堆的點數；$N_t=10$", "$N_{tY}=4$, $N_{tN}=6$: points in the yes and no piles; $N_t=10$")), "TXT", 16, False, "left", w=490, math=True)
    # bar
    sc = 400 / Ip; x0 = 0; y = 232
    f.text(x0, y - 4, T(tr("灰條 ＝ 分裂前 $I(t)$；彩色 ＝ 分完剩下的", "grey bar = $I(t)$ before; colours = what is left after")), "TXT", 16, False, "left", w=560, math=True)
    xx = x0
    for val, col in ((s1, "#5fb8a8"), (s2, "#a98bd0"), (d, "GAP")):
        f.rect(xx, y + 14, val * sc, 34, fill=col); xx += val * sc
    f.text(x0 + s1 * sc / 2, y + 66, f"0.4 × {Iy:.3f}", "TXT", 16, w=110); f.text(x0 + s1 * sc / 2, y + 86, f"= {s1:.3f}", "TXT", 16)
    f.text(x0 + s1 * sc + s2 * sc / 2, y + 66, f"0.6 × {In:.3f}", "TXT", 16, w=110); f.text(x0 + s1 * sc + s2 * sc / 2, y + 86, f"= {s2:.3f}", "TXT", 16)
    f.text(x0 + (s1 + s2) * sc + d * sc / 2, y + 66, T(tr("缺口 ＝ ΔI", "gap = ΔI")), "#7dbb8c", 16, True, w=100); f.text(x0 + (s1 + s2) * sc + d * sc / 2, y + 86, f"{d:.3f}", "#7dbb8c", 16, True)
    # formula
    f.rect(580, 226, 580, 130, fill=_tint("green", 0.12), stroke=_tint("green", 0.55), sw=1.5, radius=14)
    f.text(870, 262, "$\\Delta I=I(t)-\\frac{N_{tY}}{N_t}I(t_Y)-\\frac{N_{tN}}{N_t}I(t_N)$", "TXT", 22, w=540, math=True)
    f.text(870, 322, f"$=1.522-{s1:.3f}-{s2:.3f}={d:.3f}$", "TXT", 24, True, w=540, math=True)
    _tk(f, 0, 392, 1160, 40, T(tr("教材印 0.315；用它自己的計數重算是 0.322。", "The textbook prints 0.315; its own counts give 0.322.")))
    return f


def _fb(f, x, y, w, h):
    from figkit.parts import outline_box
    outline_box(f, x, y, w, h, "TXT", 1.5)


def tree_intro():
    from figkit.parts import NFig as _N, outline_box, header_chip, panel, chip, takeaway as _tk, tint as _tint, solid as _solid, INK
    f = _N("lec05-kp23-partition", 1160, 440, T(tr("決策樹把平面切成區域：左圖用兩個軸平行的問題把平面切成 A、B、C 三塊（x₂>0.5 切上下，x₁≤0.4 切下半左右），右圖是對應的樹；每個圓圈是節點，最下面的葉節點寫最後的類別；OBCT 每個節點只問一個特徵是否小於等於門檻。",
                                  "A decision tree cuts the plane into regions: the left figure cuts it into A, B and C with two axis-parallel questions (x₂>0.5 splits top and bottom, x₁≤0.4 splits the lower half) and the right figure is the matching tree; each circle is a node and the leaf nodes at the bottom carry the final class; an OBCT asks one feature against a threshold at each node.")))
    header_chip(f, 0, 0, 560, 34, "blue", T(tr("一連串是非題＝把平面切成區域", "A series of yes/no questions cuts the plane")), size=20)
    panel(f, 0, 38, 560, 300, "blue")
    f.rect(30, 62, 200, 200, stroke="TXT", sw=2.2); f.line(30, 148, 230, 148, "TXT", 2.4); f.line(112, 148, 112, 262, "TXT", 2.4)
    for (x, y, t, c) in ((130, 105, "A", "#6a9fd6"), (71, 205, "B", "#d97757"), (171, 205, "C", "#7dbb8c")): f.text(x, y, t, c, 26, True)
    f.text(14, 148, "0.5", "MUT", 16); f.text(112, 280, "0.4", "MUT", 16); f.text(246, 280, "x₁", "MUT", 16); f.text(30, 50, "x₂", "MUT", 16)
    def box(x, y, t, c="TXT"):
        f.rect(x - 52, y - 18, 104, 36, stroke=c, sw=2, radius=8); f.text(x, y, t, c, 16, True)
    box(400, 80, "x₂ > 0.5 ?"); box(340, 170, "x₁ ≤ 0.4 ?"); box(470, 170, T(tr("葉 A", "leaf A")), "#6a9fd6"); box(290, 260, T(tr("葉 B", "leaf B")), "#d97757"); box(400, 260, T(tr("葉 C", "leaf C")), "#7dbb8c")
    for (a, b) in (((400, 98), (340, 152)), ((400, 98), (470, 152)), ((340, 188), (290, 242)), ((340, 188), (400, 242))): f.line(a[0], a[1], b[0], b[1], "MUT", 2)
    f.text(354, 124, T(tr("否", "no")), "MUT", 16); f.text(450, 124, T(tr("是", "yes")), "MUT", 16); f.text(300, 214, T(tr("是", "yes")), "MUT", 16); f.text(390, 214, T(tr("否", "no")), "MUT", 16)
    header_chip(f, 600, 0, 560, 34, "green", T(tr("決策樹", "Decision tree")), size=20)
    panel(f, 600, 38, 560, 200, "green")
    f.text(618, 138, T(tr("一連串是非題排成一棵樹。\n每個圓圈叫節點，問一個問題；\n最下面不再問的叫葉節點，寫著最後判成哪一類。\nOBCT（普通二元分類樹）：每個節點只問\n一個特徵是否 $x_k\\leq\\alpha$？是走一邊，否走另一邊。", "A series of yes/no questions arranged as a tree.\nEach circle is a node and asks one question;\nthe bottom nodes that ask no more are leaf nodes and carry the final class.\nOBCT (ordinary binary classification tree): each node asks\nonly whether one feature $x_k\\leq\\alpha$; yes goes one way, no the other.")), "TXT", 18, False, "left", w=530, math=True, name=T(tr("term:決策樹", "term:Decision tree")))
    panel(f, 600, 248, 560, 90, "red")
    f.text(618, 293, T(tr("切出的區域邊都與座標軸平行（二維是長方形）。\n左圖三塊、右圖是對應的樹。", "Cut regions have edges parallel to the axes.\nLeft: three regions; right: the matching tree.")), "TXT", 18, False, "left", w=530)
    _tk(f, 0, 356, 1160, 44, T(tr("不畫邊界，只問一連串簡單的是非題。", "No boundary is drawn, only a series of simple yes/no questions.")))
    return f


def four_things():
    import numpy as np
    from figkit.parts import NFig as _N, header_chip, panel, takeaway as _tk, tint as _tint, solid as _solid, icon
    xs = np.array([1.0, 1.4, 2.2, 2.8, 3.5, 4.1, 5.0, 5.5, 6.2, 7.0]); mids = (xs[1:] + xs[:-1]) / 2
    f = _N("lec05-kp23-cands", 1160, 428, T(tr("建一棵樹要先決定四件事：①候選問題 ②分裂準則 ③停止規則 ④葉的類別；每個特徵的門檻只試相鄰兩值的中點：10 個值 " + "、".join(f"{v:g}" for v in xs) + " 得到 9 個門檻 " + "、".join(f"{m:.4g}" for m in mids) + "；最多 N−1 個問題。",
                                  "Four decisions to build a tree: (1) candidate questions (2) splitting criterion (3) stopping rule (4) class of a leaf; each feature's thresholds are only the midpoints of neighbouring values: 10 values " + ", ".join(f"{v:g}" for v in xs) + " give 9 thresholds " + ", ".join(f"{m:.4g}" for m in mids) + "; at most N−1 questions.")))
    items = [("blue", tr("① 候選問題", "① candidate questions"), tr("每個節點能問哪些", "questions a node may ask")), ("green", tr("② 分裂準則", "② splitting criterion"), tr("哪個問題最好", "which question is best")),
             ("orange", tr("③ 停止規則", "③ stopping rule"), tr("何時不再分", "when to stop splitting")), ("violet", tr("④ 葉的類別", "④ class of a leaf"), tr("常用多數的那一類", "usually the majority class"))]
    w = (1160 - 18 * 3) / 4
    for i, (role, h, t) in enumerate(items):
        x = i * (w + 18); header_chip(f, x, 0, w, 36, role, T(h), size=18); panel(f, x, 40, w, 56, role); f.text(x + w / 2, 68, T(t), "TXT", 18, w=w - 16)
    f.text(580, 126, T(tr("① 候選問題：每個特徵只試「相鄰兩個值的中點」", "① candidate questions: each feature only tries the midpoints of neighbouring values")), "TXT", 20, True, w=1100)
    sx = lambda v: 40 + (v - 1) / 6 * 1080; y = 290
    f.line(30, y, 1130, y, "MUT", 2)
    for v in xs: f.ellipse(sx(v), y, 16, "#6a9fd6"); f.text(sx(v), y + 28, f"{v:g}", "MUT", 16)
    for m in mids: f.line(sx(m), y - 30, sx(m), y + 30, "#7dbb8c", 3)
    for i, m in enumerate(mids): f.text(sx(m), y - 48 - (18 if i % 2 else 0), f"{m:.4g}", "#7dbb8c", 16, True)
    f.text(580, 156, T(tr("10 個不同的值（藍點）→ 9 個中點門檻（綠線）；$(1.0+1.4)/2=1.2$、$(1.4+2.2)/2=1.8$、…", "10 distinct values (blue dots) → 9 midpoint thresholds (green lines); $(1.0+1.4)/2=1.2$, $(1.4+2.2)/2=1.8$, …")), "TXT", 18, w=1100, math=True)
    f.rect(0, 338, 1160, 40, fill=_tint("blue", 0.14), radius=12)
    f.text(580, 358, T(tr("特徵 $x_k$ 在節點內有 $N_{tk}$ 個不同的值 → 最多 $N_{tk}-1$ 個問題（教材寫 $N_{tk}$ 個）", "feature $x_k$ takes $N_{tk}$ distinct values in the node → at most $N_{tk}-1$ questions (the textbook writes $N_{tk}$)")), "TXT", 20, True, w=1120, math=True)
    _tk(f, 0, 386, 1160, 36, T(tr("門檻是有限的一串，不是無限多個。", "The thresholds are a finite list, not infinitely many.")))
    return f


def bagging_full():
    import numpy as np
    from figkit.parts import NFig as _N, header_chip, panel, outline_box, icon, chip, takeaway as _tk, tint as _tint, solid as _solid, INK
    rng = np.random.default_rng(3); n = 8
    f = _N("lec05-kp23-bagsample", 1160, 432, T(tr("Bagging：從訓練集有放回地抽 B 份（和原來一樣大，有些點重複、有些沒抽到），各長一棵樹，最後多數決；圖是 8 個編號點的示意（不是實驗資料），三份樣本如第一份 1,2,2,2,5,7,7,7 漏掉 3,4,6,8；一個點沒被抽到的機率 (1−1/N)^N 趨近 0.368；隨機森林每個節點只從隨機挑的特徵裡選。",
                                  "Bagging: draw B sets from the training set with replacement (as large as the original, some points repeated and some missed), grow a tree on each and take a majority vote; the figure is an illustration with 8 numbered points (not experiment data), e.g. the first sample 1,2,2,2,5,7,7,7 misses 3,4,6,8; the chance a point is never drawn, (1−1/N)^N, approaches 0.368; a random forest lets each node choose among randomly picked features only.")))
    header_chip(f, 0, 0, 560, 34, "blue", T(tr("有放回地抽樣（示意，不是實驗資料）", "Drawing with replacement (illustration, not experiment data)")), size=18)
    panel(f, 0, 38, 560, 330, "blue")
    def row(y, nums):
        for k, v in enumerate(nums):
            cx = 130 + 36 * k; f.rect(cx - 14, y - 14, 28, 28, stroke="TXT", sw=1.8, radius=14); f.text(cx, y, str(v), "TXT", 16, True)
    f.text(60, 72, T(tr("原本", "original")), "TXT", 16, True); row(72, range(1, 9))
    for i in range(3):
        sm = sorted(rng.integers(1, n + 1, n).tolist()); y = 124 + i * 52
        f.text(60, y, T(tr(f"樣本 {i+1}", f"sample {i+1}")), "TXT", 16, True); row(y, sm)
        miss = [v for v in range(1, n + 1) if v not in sm]
        f.text(424, y, T(tr("沒抽到：", "missed: ")) + (" ".join(map(str, miss)) if miss else T(tr("無", "none"))), "#e07a70", 16, True, "left")
    f.text(280, 270, T(tr("每份樣本長一棵樹，三棵樹投票：", "each sample grows a tree and the three trees vote:")), "TXT", 16, True, w=520)
    for i, k in enumerate(("circle", "square", "circle")):
        x = 120 + i * 150; f.text(x, 308, T(tr(f"樹 {i+1}", f"tree {i+1}")), "TXT", 16, True); icon(f, k, x + 50, 308)
    chip(f, 60, 326, 440, 30, "green", T(tr("多數決：圓形 2 票對 1 票，圓形勝", "majority: circle wins 2 votes to 1")), size=18, bold=True)
    header_chip(f, 600, 0, 560, 34, "green", "Bagging", size=20)
    panel(f, 600, 38, 560, 206, "green")
    f.text(618, 120, T(tr("Bagging：從訓練集有放回地抽 $B$ 份\n（和原來一樣大，有些點重複、有些沒抽到），\n各長一棵樹，最後多數決。\n隨機森林：再加一招，每個節點只從\n隨機挑的特徵裡選（這裡每次只看 1 個）。", "Bagging: draw $B$ sets from the training set with replacement\n(as large as the original, some points repeated, some missed),\ngrow a tree on each and take a majority vote.\nRandom forest: one more trick, each node chooses only among\nrandomly picked features (here just 1 at a time).")), "TXT", 18, False, "left", w=530, math=True, name=T(tr("term:Bagging", "term:Bagging")))
    panel(f, 600, 256, 560, 112, "orange")
    f.text(880, 312, T(tr("一個點一次都沒被抽到的機率\n$(1-1/N)^N\\rightarrow 0.368$（約 37% 的點沒被抽到）", "chance that a given point is never drawn\n$(1-1/N)^N\\rightarrow 0.368$ (about 37% of the points are missed)")), "TXT", 18, w=530, math=True)
    _tk(f, 0, 384, 1160, 44, T(tr("每份樣本都不一樣，各樹的錯也不一樣，投票時互相抵消。", "Every sample differs, so the trees err differently and the vote cancels the errors.")))
    return f


def bagres_plot():
    import json, os
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    import numpy as np
    N = _nums(); cp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kp23_bagres_cache.json")
    if os.path.exists(cp): single, bag, rf = json.load(open(cp))
    else:
        Xte, yte = N.moons2(1000, 2); single, bag, rf = [], [], []
        for s in range(20):
            X, y = N.moons2(100, 100 + s)
            single.append(DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y).score(Xte, yte))
            bag.append(BaggingClassifier(DecisionTreeClassifier(criterion="entropy"), n_estimators=25, random_state=0).fit(X, y).score(Xte, yte))
            rf.append(RandomForestClassifier(n_estimators=25, criterion="entropy", max_features=1, random_state=0).fit(X, y).score(Xte, yte))
        json.dump([single, bag, rf], open(cp, "w"))
    rr = np.random.default_rng(1); rows = [(tr("單棵樹", "single tree"), single, "red", 2.0), ("Bagging", bag, "green", 1.0), (tr("隨機森林", "random forest"), rf, "blue", 0.0)]
    ms = lambda v: (float(np.mean(v)) * 100, float(np.std(v)) * 100)
    series = []
    for nm, v, role, yc in rows:
        series.append({"kind": "dots", "points": [(a * 100, yc + float(rr.uniform(-0.28, 0.28))) for a in v], "role": role, "d": 11})
        series.append({"kind": "seg", "a": (ms(v)[0], yc - 0.42), "b": (ms(v)[0], yc + 0.42), "role": role, "width": 5})
    sm = [ms(v) for _, v, _, _ in rows]
    spec = {"id": "lec05-kp23-bagres",
            "alt": tr(f"20 份訓練集（種子 100 到 119）在 2000 個新點上的準確率，每點一份、直條是平均：單棵樹 {sm[0][0]:.1f}±{sm[0][1]:.1f}%、Bagging {sm[1][0]:.1f}±{sm[1][1]:.1f}%、隨機森林 {sm[2][0]:.1f}±{sm[2][1]:.1f}%。",
                      f"Accuracy of 20 training sets (seeds 100 to 119) on 2000 new points, one dot each, the bar is the mean: single tree {sm[0][0]:.1f}±{sm[0][1]:.1f}%, bagging {sm[1][0]:.1f}±{sm[1][1]:.1f}%, random forest {sm[2][0]:.1f}±{sm[2][1]:.1f}%."),
            "plot": {"x": [80, 90], "y": [-0.6, 2.6], "x_ticks": [80, 82, 84, 86, 88, 90], "y_ticks": [], "y_names": {"2": tr("單棵樹", "single tree"), "1": "Bagging", "0": tr("隨機森林", "random forest")}, "x_label": tr("新資料準確率（%）", "accuracy on new data (%)"), "w": 560, "h": 300},
            "series": series,
            "side": [{"title": tr("怎麼讀", "How to read"), "role": "blue", "text": tr("20 份不同的訓練集（種子 100–119）\n每點＝一份在 2000 個新點的準確率\n直條＝20 份的平均；每個森林 $B=25$ 棵", "20 different training sets (seeds 100-119)\neach dot = accuracy on 2000 new points\nbar = mean of the 20; every forest $B=25$ trees")},
                     {"title": tr("結果（平均 ± 標準差）", "Result (mean ± sd)"), "role": "green", "text": tr(f"單棵樹 {sm[0][0]:.1f} ± {sm[0][1]:.1f}%\nBagging {sm[1][0]:.1f} ± {sm[1][1]:.1f}%\n隨機森林 {sm[2][0]:.1f} ± {sm[2][1]:.1f}%", f"single tree {sm[0][0]:.1f} ± {sm[0][1]:.1f}%\nbagging {sm[1][0]:.1f} ± {sm[1][1]:.1f}%\nrandom forest {sm[2][0]:.1f} ± {sm[2][1]:.1f}%")}],
            "takeaway": tr("多棵樹的錯不同，投票時互相抵消：平均升高、散布變小。", "The errors of many trees differ and cancel in the vote: higher mean, smaller spread.")}
    return "plot_xy", spec


def practice_page():
    from figkit.parts import NFig as _N, header_chip, panel, outline_box, icon, chip, takeaway as _tk, tint as _tint, solid as _solid
    f = _N("lec05-kp23-practice", 1160, 420, T(tr("練習：節點 t=(4,4)（4 個第一類、4 個第二類），N_t=8，I(t)=1；分裂 A：是 (3,1)、否 (1,3)；分裂 B：是 (2,2)、否 (2,2)；已知 I(3,1)=0.811。求兩個分裂的 ΔI，哪個較好？",
                                  "Exercise: node t=(4,4) (4 of class 1, 4 of class 2), N_t=8, I(t)=1; split A: yes (3,1), no (1,3); split B: yes (2,2), no (2,2); I(3,1)=0.811 is given. Find ΔI of both splits; which is better?")))
    def box(x, y, c1, c2):
        w = max(c1, c2) * 30 + 20; outline_box(f, x, y, w, 76, "TXT", 1.5)
        for k in range(c1): icon(f, "circle", x + 10 + 15 + k * 30 - 10 + 10, y + 24)
        for k in range(c2): icon(f, "square", x + 10 + 15 + k * 30 - 10 + 10, y + 54)
        return w
    outline_box(f, 460, 0, 240, 76, "TXT", 1.5)
    for k in range(4): icon(f, "circle", 484 + k * 30, 24); icon(f, "square", 484 + k * 30, 54)
    f.text(580, 100, T(tr("節點 $t=(4,4)$：$N_t=8$，$I(t)=1$", "node $t=(4,4)$: $N_t=8$, $I(t)=1$")), "TXT", 18, True, w=420, math=True)
    for x0, role, hd, ys, ns in ((0, "red", tr("分裂 A", "Split A"), ((3, 1), (1, 3)), "yes (3,1) | no (1,3)"), (600, "blue", tr("分裂 B", "Split B"), ((2, 2), (2, 2)), "yes (2,2) | no (2,2)")):
        header_chip(f, x0, 126, 560, 34, role, T(hd), size=20); panel(f, x0, 164, 560, 130, role)
        bx = x0 + 70
        for c1, c2 in ys:
            w = box(bx, 180, c1, c2); f.line(x0 + 280, 168, bx + w / 2, 180, "MUT", 2); bx += w + 120
        f.text(x0 + 280, 280, T(tr("是／否 兩堆各 4 點", "yes and no piles of 4 points each")), "TXT", 16, w=500)
    chip(f, 0, 312, 380, 40, "yellow", T(tr("① 分裂 A 的 ΔI ＝ ？", "① ΔI of split A = ?")), size=20, bold=True)
    chip(f, 390, 312, 380, 40, "yellow", T(tr("② 分裂 B 的 ΔI ＝ ？", "② ΔI of split B = ?")), size=20, bold=True)
    chip(f, 780, 312, 380, 40, "yellow", T(tr("③ 哪個分裂比較好？", "③ which split is better?")), size=20, bold=True)
    f.text(580, 378, T(tr("已知 $I(3,1)=0.811$；$\\Delta I=I(t)-\\frac{N_{tY}}{N_t}I(t_Y)-\\frac{N_{tN}}{N_t}I(t_N)$", "given $I(3,1)=0.811$; $\\Delta I=I(t)-\\frac{N_{tY}}{N_t}I(t_Y)-\\frac{N_{tN}}{N_t}I(t_N)$")), "TXT", 20, w=1100, math=True)
    return f


def solution_steps():
    a = 1 - 0.5 * H((3, 1)) - 0.5 * H((1, 3)); b = 1 - 0.5 * 1 - 0.5 * 1
    spec = {"id": "lec05-kp23-practice-sol",
            "alt": tr(f"解答：分裂 A 兩堆各 4 點、I=0.811，ΔI=1−(4/8)0.811−(4/8)0.811={a:.3f}；分裂 B 兩堆 (2,2)、I=1，ΔI=1−(4/8)·1−(4/8)·1={b:g}；A 比較好；若分成 (4,0) 與 (0,4)，ΔI=1 最大。",
                      f"Solution: split A has two piles of 4 points, I=0.811, ΔI=1−(4/8)0.811−(4/8)0.811={a:.3f}; split B has piles (2,2), I=1, ΔI=1−(4/8)·1−(4/8)·1={b:g}; A is better; splitting into (4,0) and (0,4) gives the maximum ΔI=1."),
            "steps": [{"role": "red", "title": tr("① 分裂 A", "① split A"), "lines": ["$I(3,1)=I(1,3)=0.811$", "$1-\\frac{4}{8}\\cdot0.811-\\frac{4}{8}\\cdot0.811$"], "result": f"$\\Delta I={a:.3f}$"},
                      {"role": "blue", "title": tr("② 分裂 B", "② split B"), "lines": ["$I(2,2)=1$", "$1-\\frac{4}{8}\\cdot1-\\frac{4}{8}\\cdot1$"], "result": f"$\\Delta I={b:g}$"},
                      {"role": "green", "title": tr("③ 核對", "③ check"), "lines": [tr("A 比較好；", "A is better;"), tr("B 完全沒有降低不純度", "B lowers the impurity by nothing"), tr("若分成 (4,0)、(0,4)：ΔI = 1，最大", "(4,0)|(0,4) split: ΔI = 1, max")], "result": tr("選 A", "pick A")}],
            "takeaway": tr("ΔI 越大，分裂後整體改善越多。", "The larger ΔI, the more the split improves overall.")}
    return "steps_flow", spec
