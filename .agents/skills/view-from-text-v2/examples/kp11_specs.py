"""KP11 (pruning, regularization, simulation evidence) -- fourth sample of view-from-text-v2. Plan: kp11_plan.json.
Every function returns (layout_name, spec) for layouts.build(), or an NFig for the freeform pages."""
import os, sys
from math import sqrt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, takeaway, tint, solid, INK


def tr(zh, en): return {"zh": zh, "en": en}


def curvature():
    A = [(-0.26 + 0.26 * i / 40, 0.5 * 100 * (-0.26 + 0.26 * i / 40) ** 2) for i in range(41)]
    A = [p for p in A if p[1] <= 3.2]
    B = [(-1.2 + 1.2 * i / 40, 0.5 * 0.02 * (-1.2 + 1.2 * i / 40) ** 2) for i in range(41)]
    cA, cB = 0.5 * 100 * 0.2 ** 2, 0.5 * 0.02 * 1.0 ** 2
    spec = {
        "id": "lec05-kp11-curvature",
        "alt": tr(f"局部二次損失曲線：A 的權重 0.2、曲率 100，曲線陡，刪除（修改量 −0.2）的代價 {cA:g}；B 的權重 1、曲率 0.02，曲線平，刪除（修改量 −1）的代價 {cB:g}。權重小的 A 刪除代價反而高。",
                  f"Local quadratic loss curves: A has weight 0.2 and curvature 100, a steep curve, and removing it (change -0.2) costs {cA:g}; B has weight 1 and curvature 0.02, a flat curve, and removing it (change -1) costs {cB:g}. The smaller weight A costs more to remove."),
        "plot": {"x": [-1.25, 0.12], "y": [0, 3.2], "x_ticks": [-1, -0.5, 0], "y_ticks": [0, 1, 2, 3], "x_label": tr("修改量 δw（局部二次示意）", "change δw (local quadratic sketch)"), "y_label": tr("代價增加 ΔJ", "cost increase ΔJ"), "w": 440, "h": 300, "legend_pos": "left"},
        "series": [{"kind": "line", "points": A, "role": "red", "width": 4, "name": tr("A：權重 0.2、曲率 100", "A: w 0.2, curv. 100")},
                   {"kind": "line", "points": B, "role": "blue", "width": 4, "name": tr("B：權重 1、曲率 0.02", "B: w 1, curv. 0.02")},
                   {"kind": "dots", "points": [(-0.2, cA)], "role": "red", "d": 16}, {"kind": "dots", "points": [(-1.0, cB)], "role": "blue", "d": 16}],
        "notes": [{"x": -0.5, "y": cA, "text": tr(f"A 刪除：{cA:g}", f"A removed: {cA:g}"), "ha": "center"},
                  {"x": -1.0, "y": 0.55, "text": tr(f"B 刪除：{cB:g}", f"B removed: {cB:g}"), "ha": "center"}],
        "side": [{"title": tr("剪枝", "Pruning"), "role": "green", "name": tr("term:剪枝", "term:Pruning"), "h": 1.0, "text": tr("移除部分連線或單元，\n減少模型可調的部分；\n刪除後仍要檢查表現。", "remove some connections or units\nto cut what the model can adjust;\nthe result must still be checked.")},
                 {"title": tr("曲率", "Curvature"), "role": "blue", "name": tr("term:曲率", "term:Curvature"), "h": 1.0, "text": tr("二階導數：損失的斜率\n改變得多快；曲率大，\n走一小步代價就變很多。", "second derivative: how fast the loss slope\nchanges; large curvature means a small\nstep changes the cost a lot.")},
                 {"title": tr("海森矩陣", "Hessian"), "role": "orange", "name": tr("term:海森矩陣", "term:Hessian"), "h": 1.0, "text": tr("所有權重二階偏導數排成的矩陣 $H$；\n$H_{ii}$ 是第 $i$ 個方向的曲率。", "matrix of all second partial derivatives;\n$H_{ii}$ = curvature in direction $i$.")}],
        "takeaway": tr("刪除代價由曲率與權重一起決定：權重小的 A 代價 2，權重大的 B 只有 0.01。", "The removal cost depends on curvature and weight together: the small weight A costs 2, the large weight B only 0.01."),
    }
    return "plot_xy", spec


def saliency():
    f = NFig("lec05-kp11-saliency", 1160, 460, T(tr("顯著度的推導四步：泰勒展開 ΔJ≈Σg·δw+½ΣH·δw·δw；接近極小值 g≈0；對角近似；代入 δw=−w 得 s=½H_ii w_i²。例：A 為 2、B 為 0.01，相差 200 倍，依局部估計先刪 B。估計的條件：尚未近極小值、刪太多、多權重耦合、負曲率。",
                                              "Four steps to the saliency: Taylor expansion ΔJ≈Σg·δw+½ΣH·δw·δw; near a minimum g≈0; the diagonal approximation; substituting δw=−w gives s=½H_ii w_i². Example: A is 2, B is 0.01, a factor of 200, so by the local estimate remove B first. Conditions: not yet near the minimum, deleting too much, coupled weights, negative curvature.")))
    steps = [("blue", "$\\Delta J\\approx\\sum_i g_i\\delta w_i+\\frac{1}{2}\\sum_{i,j}H_{ij}\\delta w_i\\delta w_j$", tr("泰勒展開：用斜率 $g$ 與曲率 $H$\n近似代價的變化", "Taylor expansion: slope $g$ and\ncurvature $H$ approximate the change"), "term:泰勒展開", "term:Taylor expansion"),
             ("green", "$g_i\\approx0$", tr("接近極小值，斜率項消失", "near a minimum the slope term vanishes"), None, None),
             ("orange", "$\\Delta J\\approx\\frac{1}{2}\\sum_i H_{ii}(\\delta w_i)^2$", tr("對角近似：忽略不同權重的交互項", "diagonal approximation: drop the cross terms"), None, None),
             ("red", "$s_i=\\frac{1}{2}H_{ii}w_i^2$", tr("刪除就是 $\\delta w_i=-w_i$；\n$s_i$ 叫顯著度，是刪除代價的估計", "removal: $\\delta w_i=-w_i$;\n$s_i$ is the saliency, a cost estimate"), "term:顯著度", "term:Saliency")]
    for k, (role, tex, why, zh_nm, en_nm) in enumerate(steps):
        y = 6 + k * 70
        chip(f, 0, y + 8, 40, 40, role, str(k + 1), size=22, bold=True, dark=True)
        f.rect(52, y, 400, 56, fill=tint(role, 0.2), stroke=solid(role), sw=2, radius=12, text=tex, tcolor="TXT", size=20, bold=False, math=True)
        nm = T(tr(zh_nm, en_nm), None) if zh_nm else None
        f.text(466, y + 28, T(why), "TXT", 16, False, "left", w=270, name=nm, math="$" in T(why))
    # the worked example on a log scale
    bx = 760
    f.text(bx + 190, 14, T(tr("A、B 兩條連線（對數刻度）", "Two connections A and B (log scale)")), "TXT", 18, True, w=380)
    import math
    sc = lambda v: (math.log10(v) + 3) / 4 * 340
    for i, (nm, tex, v, role) in enumerate((("A", "$\\frac{1}{2}\\times100\\times0.2^2=2$", 2, "red"), ("B", "$\\frac{1}{2}\\times0.02\\times1^2=0.01$", 0.01, "blue"))):
        y = 50 + i * 100
        f.text(bx, y + 12, nm, "TXT", 22, True, "left", w=30)
        f.rect(bx + 30, y, max(sc(v), 6), 34, fill=solid(role), radius=6)
        f.text(bx + 30, y + 58, tex, "TXT", 20, False, "left", w=360, math=True)
    chip(f, bx + 8, 258, 380, 40, "yellow", T(tr("2 ÷ 0.01 = 200：依局部估計先刪 B", "2 ÷ 0.01 = 200 → remove B first")), size=18, bold=True, dark=True)
    # conditions
    f.text(0, 304, T(tr("這個估計成立的條件", "When this estimate holds")), "TXT", 18, True, "left", w=400)
    conds = [tr("尚未接近極小值：一階項未必小", "far from a minimum: slope term may be large"), tr("刪除幅度太大：高階項不能忽略", "removing a lot: higher orders matter"),
             tr("同時刪多個：權重之間會耦合", "removing several: weights couple"), tr("負曲率：不能把負數當成安全", "negative curvature is not safe")]
    for k, c in enumerate(conds):
        x = k * 292
        f.rect(x, 322, 276, 58, fill=tint("grey", 0.14), stroke=tint("grey", 0.6), sw=1.5, radius=10)
        f.text(x + 138, 351, T(c), "TXT", 16, False, w=262)
    takeaway(f, 0, 396, 1160, 44, T(tr("顯著度是不真的刪除就能算的估計，不是實測損失：要小步刪除、重新量測。", "The saliency needs no deletion but is an estimate, not a measured loss: delete in small steps.")))
    return f


def prune_flow():
    spec = {
        "id": "lec05-kp11-flow",
        "alt": tr("剪枝流程四步：先訓練到合適的位置；估每條連線的顯著度；刪少量低顯著度的連線；再訓練並用驗證資料檢查。依驗證表現、允許的損失增加或資源預算停止，刪除後的小網路不保證更準。",
                  "Four pruning steps: train to a suitable point; estimate each connection's saliency; delete a few low-saliency connections; retrain and check on validation data. Stop by validation performance, an allowed loss increase or a resource budget; the smaller network is not guaranteed to be more accurate."),
        "steps": [{"role": "blue", "title": tr("① 訓練", "① Train"), "lines": [tr("先訓練到", "train to a"), tr("合適的位置", "suitable point")]},
                  {"role": "green", "title": tr("② 估顯著度", "② Estimate"), "lines": [tr("每條連線算", "for every connection"), "$s_i=\\frac{1}{2}H_{ii}w_i^2$"]},
                  {"role": "orange", "title": tr("③ 少量刪除", "③ Delete a few"), "lines": [tr("只刪顯著度", "only the lowest"), tr("最低的少數條", "saliency ones")]},
                  {"role": "red", "title": tr("④ 再訓練並檢查", "④ Retrain, check"), "lines": [tr("剩下的權重再訓練", "retrain what is left"), tr("用驗證資料檢查", "check on validation data")], "result": tr("回到 ②", "back to ②")}],
        "takeaway": tr("依驗證表現、允許的損失增加或資源預算停止；排序不能永久當成真理。", "Stop by validation performance, an allowed loss increase or a resource budget; the ranking is never permanent truth."),
    }
    return "steps_flow", spec


def regularization():
    f = NFig("lec05-kp11-reg", 1160, 460, T(tr("正則化：J_total=J_data+αP。α=0.1：模型 A 資料誤差 0.10、懲罰 P=4，總代價 0.10+0.1×4=0.50；模型 B 資料誤差 0.20、P=1，總代價 0.20+0.1×1=0.30，選 B。α=0：只看資料誤差 0.10 對 0.20，選 A。α 要用驗證資料選；總代價較小不等於測試誤差必然較小。",
                                              "Regularization: J_total=J_data+αP. With α=0.1: model A has data error 0.10 and penalty P=4, total 0.10+0.1×4=0.50; model B has data error 0.20 and P=1, total 0.20+0.1×1=0.30, so B is chosen. With α=0 only the data error counts, 0.10 against 0.20, so A is chosen. Choose α on validation data; a smaller total cost does not guarantee a smaller test error.")))
    f.rect(0, 0, 1160, 64, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(20, 32, T(tr("正則化：在資料誤差之外，加入對模型的限制或偏好；它改變我們願意接受的解。", "Regularization: besides the data error, add a limit or preference on the model; it changes which solutions we accept.")), "TXT", 17, False, "left", w=560, name=T(tr("term:正則化", "term:Regularization")))
    f.text(880, 32, "$J_{\\mathrm{total}}=J_{\\mathrm{data}}+\\alpha P(\\mathbf{w})$", "TXT", 26, True, w=500, math=True)
    sc = lambda v: v / 0.6 * 300
    for p, (x0, hd, al, rows, pick) in enumerate((
            (0, tr("α = 0.1：總代價 = 資料誤差 + 0.1 × 懲罰 P", "α = 0.1: total = data error + 0.1 × penalty P"), 0.1, (("A", 0.10, 4), ("B", 0.20, 1)), "B"),
            (600, tr("α = 0：只看資料誤差", "α = 0: data error only"), 0.0, (("A", 0.10, 4), ("B", 0.20, 1)), "A"))):
        header_chip(f, x0, 82, 560, 34, "green" if p == 0 else "grey", T(hd), size=18)
        panel(f, x0, 120, 560, 190, "green" if p == 0 else "grey")
        for i, (nm, d, P) in enumerate(rows):
            y = 142 + i * 78
            pen = al * P; tot = d + pen
            f.text(x0 + 20, y + 20, nm, "TXT", 26, True, "left", w=40)
            f.rect(x0 + 60, y, sc(d), 40, fill=solid("blue"), radius=4)
            if pen > 0: f.rect(x0 + 60 + sc(d), y, sc(pen), 40, fill=solid("orange"), radius=4)
            f.text(x0 + 60 + sc(tot) + 12, y + 20, f"{tot:.2f}", "TXT", 24, True, "left", w=80)
            f.text(x0 + 60, y + 58, T(tr(f"資料 {d:.2f}", f"data {d:.2f}")) + (T(tr(f" ＋ 0.1×{P} = {pen:.2f}", f" + 0.1×{P} = {pen:.2f}")) if pen > 0 else ""), "MUT", 16, False, "left", w=330)
            if nm == pick:
                chip(f, x0 + 450, y + 4, 96, 34, "yellow", T(tr(f"選 {nm}", f"pick {nm}")), size=20, bold=True, dark=True)
    f.rect(50, 322, 14, 14, fill=solid("blue")); f.text(72, 329, T(tr("資料誤差", "data error")), "TXT", 16, False, "left", w=120)
    f.rect(220, 322, 14, 14, fill=solid("orange")); f.text(242, 329, T(tr("α × 懲罰 P（P：A=4、B=1）", "α × penalty P (P: A=4, B=1)")), "TXT", 16, False, "left", w=360)
    f.text(580, 358, T(tr("α ≥ 0 是懲罰的比重（與動量的 α 不同）；資料誤差用總和或平均時，同一個 α 的作用不同，不能跨設定照抄。", "α ≥ 0 weighs the penalty (not the momentum α); summed vs averaged data error changes its effect, so do not copy α.")), "MUT", 16, w=1120)
    takeaway(f, 0, 388, 1160, 44, T(tr("α 要用驗證資料選；總代價較小不等於測試誤差必然較小。", "Choose α on validation data; a smaller total cost does not guarantee a smaller test error.")))
    return f


def saturating():
    xs = [i * 0.05 for i in range(0, 81)]
    rho = lambda w: w * w / (1 + w * w); drho = lambda w: 2 * w / (1 + w * w) ** 2
    spec = {
        "id": "lec05-kp11-rho",
        "alt": tr(f"飽和懲罰 ρ(w)=w²/(τ²+w²)，τ=1：w=0.2、1、3 時約 {rho(0.2):.4f}、{rho(1):g}、{rho(3):g}，大權重時接近 1；導數 ρ′(w)=2τ²w/(τ²+w²)² 在 w=0 時為 0，所以更新不保證恰好到零，要另設門檻 ε 刪除小權重。",
                  f"Saturating penalty ρ(w)=w²/(τ²+w²) with τ=1: at w=0.2, 1, 3 it is about {rho(0.2):.4f}, {rho(1):g}, {rho(3):g} and it approaches 1 for large weights; its derivative ρ′(w)=2τ²w/(τ²+w²)² is 0 at w=0, so updates do not reach exactly zero and a threshold ε is used to delete small weights."),
        "plot": {"x": [0, 4], "y": [0, 1.05], "x_ticks": [0, 1, 2, 3, 4], "y_ticks": [0, 0.5, 1], "x_label": tr("權重大小 |w|（τ = 1）", "weight size |w| (τ = 1)"), "y_label": "", "w": 400, "h": 300, "legend_pos": "left"},
        "series": [{"kind": "line", "points": [(x, rho(x)) for x in xs], "role": "green", "width": 4, "name": tr("懲罰 ρ(w)", "penalty ρ(w)")},
                   {"kind": "line", "points": [(x, drho(x)) for x in xs], "role": "orange", "width": 4, "name": tr("導數 ρ′(w)", "derivative ρ′(w)")},
                   {"kind": "dots", "points": [(0.2, rho(0.2)), (1, rho(1)), (3, rho(3))], "role": "green", "d": 16}],
        "notes": [{"x": 0.2, "y": rho(0.2) + 0.1, "text": f"≈{rho(0.2):.4f}", "ha": "left"}, {"x": 1.0, "y": rho(1) + 0.1, "text": "0.5"}, {"x": 3.0, "y": rho(3) - 0.1, "text": "0.9"}],
        "side": [{"title": tr("飽和懲罰", "Saturating penalty"), "role": "green", "name": tr("term:飽和懲罰", "term:Saturating penalty"), "h": 1.0, "text": tr("$\\rho(w)=w^2/(\\tau^2+w^2)$，$\\tau>0$ 是尺度；很大時趨近 1\n（飽和）；總懲罰 $P=\\sum_i\\rho(w_i)$。", "$\\rho(w)=w^2/(\\tau^2+w^2)$, scale $\\tau>0$; tends to 1 for large\n$w$ (saturates); total penalty $P=\\sum_i\\rho(w_i)$.")},
                 {"title": tr("導數在 0 為 0", "Derivative is 0 at 0"), "role": "orange", "h": 1.0, "text": tr("$\\rho'(w)=2\\tau^2w/(\\tau^2+w^2)^2$；$w=0$ 時為 0，\n有限步更新不保證恰好到零。", "$\\rho'(w)=2\\tau^2w/(\\tau^2+w^2)^2$; it is 0 at $w=0$,\nso finite updates do not reach exact zero.")},
                 {"title": tr("權重消除", "Weight elimination"), "role": "blue", "name": tr("term:權重消除", "term:Weight elimination"), "h": 1.0, "text": tr("訓練時促進縮小，再定期把 $|w_i|<\\varepsilon$ 的\n權重設零或刪除；$\\varepsilon$ 是門檻，$\\tau$ 是尺度。", "shrink in training, then periodically zero the\nweights with $|w_i|<\\varepsilon$; $\\varepsilon$ is the threshold, $\\tau$ the scale.")}],
        "takeaway": tr("平滑的懲罰只把小權重壓小，要靠門檻 ε 才真正刪除。", "A smooth penalty only squeezes small weights; the threshold ε does the actual deleting."),
    }
    return "plot_xy", spec


def operations():
    spec = {
        "id": "lec05-kp11-ops",
        "alt": tr("三種不同的操作要分別記錄：刪連線（把一條權重設成零）、刪單元（整個單元連同連線移除，要檢查整條訊號路徑）、縮小數值（權重或單元輸出變小但仍在，可能被很大的下一層權重放大）。來源的複雜度項含 A³√(log d/N)：A 加倍變 2³=8 倍，只是量級摘要，不是分類錯誤率。",
                  "Three different operations are recorded separately: deleting a connection (set one weight to zero), deleting a unit (remove the unit with its connections; check the whole signal path), shrinking a value (a weight or unit output gets smaller but stays, and may be amplified by a large next-layer weight). The source's complexity term contains A³√(log d/N): doubling A gives 2³=8 times, an order-of-magnitude summary, not a classification error rate."),
        "columns": [
            {"role": "blue", "header": tr("刪連線", "Delete a connection"), "items": [{"label": tr("做法", "how"), "lines": [tr("把一條權重", "set one weight"), tr("設成零", "to zero")]}, {"label": tr("要檢查", "check"), "lines": [tr("刪除後的資料表現", "the data performance"), tr("（顯著度只是估計）", "(saliency: an estimate)")]}]},
            {"role": "green", "header": tr("刪單元", "Delete a unit"), "items": [{"label": tr("做法", "how"), "lines": [tr("整個單元連同", "remove the unit"), tr("它的連線一起移除", "with its connections")]}, {"label": tr("要檢查", "check"), "lines": [tr("整條訊號路徑，", "the whole signal path,"), tr("不只一條連線", "not one connection")]}]},
            {"role": "orange", "header": tr("縮小數值", "Shrink a value"), "items": [{"label": tr("做法", "how"), "lines": [tr("權重或單元輸出", "a weight or output"), tr("變小，但仍在", "gets smaller but stays")]}, {"label": tr("風險", "risk"), "lines": [tr("小輸出可能被很大的", "a small output may be"), tr("下一層權重放大", "amplified downstream")]}]},
            {"role": "red", "header": tr("來源的複雜度項", "Source's complexity"), "items": [{"label": tr("內容", "term"), "lines": ["$A^3\\sqrt{\\log d/N}$", tr("$A$ 單元權重絕對值和上界", "$A$: bound on sum of |weights|"), tr("$d$ 輸入維度、$N$ 樣本數", "$d$ inputs, $N$ samples")]}, {"label": tr("例", "example"), "lines": [tr("$A$ 加倍 → $2^3=8$ 倍", "doubling $A$ → $2^3=8$ times"), tr("不是分類錯誤率", "not an error rate")]}]}],
        "takeaway": tr("三種操作分開記錄；複雜度項的 A³ 只是量級摘要，不能當成錯誤率。", "Record the three separately; A³ is an order-of-magnitude summary, not an error rate."),
    }
    return "columns_text", spec


def cascade():
    f = NFig("lec05-kp11-cascade", 1160, 470, T(tr("級聯相關法：殘差 e=t−ŷ。四步：①算殘差；②階段一：訓練新單元的輸入側，使它的輸出與殘差的相關分數 S 大；③凍結輸入側權重；④階段二：重訓輸出側降低平方誤差。教學分數 S=|Σ(v−v̄)(e−ē)|：e=(−1,1)，候選 v=(−1,1) 得 2，常數候選 v=(1,1) 得 0。S 沒有除標準差，不是 Pearson 係數。",
                                              "Cascade correlation: residual e=t−ŷ. Four steps: 1 compute the residual; 2 stage one: train the input side of a new unit so the correlation score S between its output and the residual is large; 3 freeze the input-side weights; 4 stage two: retrain the output side to reduce the squared error. Teaching score S=|Σ(v−v̄)(e−ē)|: with e=(−1,1), the candidate v=(−1,1) gives 2 and the constant candidate v=(1,1) gives 0. S is not divided by the standard deviations, so it is not the Pearson coefficient.")))
    f.rect(0, 0, 570, 70, fill=tint("blue", 0.12), stroke=tint("blue", 0.55), sw=1.5, radius=12)
    f.text(14, 35, T(tr("殘差：真實目標減去目前預測，$e_n=t_n-\\hat y_n$，$n$ 是樣本編號。", "Residual: target minus prediction, $e_n=t_n-\\hat y_n$, $n$ = sample.")), "TXT", 17, False, "left", w=545, math=True, name=T(tr("term:殘差", "term:Residual")))
    f.rect(590, 0, 570, 70, fill=tint("green", 0.12), stroke=tint("green", 0.55), sw=1.5, radius=12)
    f.text(604, 35, T(tr("級聯相關法：每次增加一個能回應目前殘差的隱藏單元，再重訓輸出。", "Cascade correlation: add one hidden unit that responds to the current residual, then retrain the output.")), "TXT", 17, False, "left", w=545, name=T(tr("term:級聯相關法", "term:Cascade correlation")))
    steps = [("blue", tr("① 算殘差", "① Residual"), "$e_n=t_n-\\hat y_n$"), ("orange", tr("② 階段一", "② Stage one"), tr("訓練新單元輸入側，使分數 $S$ 大", "train its input side: $S$ large")),
             ("grey", tr("③ 凍結", "③ Freeze"), tr("凍結新單元的輸入側權重", "freeze the input-side weights")), ("green", tr("④ 階段二", "④ Stage two"), tr("重訓輸出側，降低平方誤差", "retrain the output side to cut the squared error"))]
    w = (1160 - 3 * 36) / 4
    for k, (role, hd, tx) in enumerate(steps):
        x = k * (w + 36)
        header_chip(f, x, 96, w, 34, role, T(hd), size=18)
        panel(f, x, 134, w, 84, role)
        f.text(x + w / 2, 176, T(tx), "TXT", 16, False, w=w - 20, math="$" in T(tx))
        if k < 3: f.text(x + w + 18, 176, "→", "TXT", 26, True, w=36)
    f.text(0, 240, T(tr("教學分數（單輸出）$S=\\left|\\sum_n(v_n-\\bar v)(e_n-\\bar e)\\right|$，橫線是樣本平均；$e=(-1,1)$", "Teaching score (one output) $S=\\left|\\sum_n(v_n-\\bar v)(e_n-\\bar e)\\right|$, bars are sample means; $e=(-1,1)$")), "TXT", 18, False, "left", w=1160, math=True)
    for k, (role, hd, tex, res) in enumerate(((("green"), tr("候選 $v=(-1,1)$", "candidate $v=(-1,1)$"), "$S=|{-1}\\times(-1)+1\\times1|$", "$S=2$"),
                                              (("red"), tr("常數候選 $v=(1,1)$", "constant candidate $v=(1,1)$"), tr("中心化後全是 0", "all 0 once centred"), "$S=0$"))):
        x = k * 590
        panel(f, x, 270, 570, 90, role)
        f.text(x + 16, 292, T(hd), "TXT", 18, True, "left", w=540, math=True)
        f.text(x + 16, 330, T(tex), "TXT", 20, False, "left", w=380, math=True)
        chip(f, x + 440, 308, 114, 44, role, T(res), size=24, bold=True, dark=True, math=True)
    takeaway(f, 0, 378, 1160, 44, T(tr("兩個階段分開：先找與殘差相關的候選並凍結，再讓輸出側降低誤差。S 沒除標準差，不是 Pearson 係數。", "Keep the stages apart: freeze the best candidate, then retrain the output. S is not Pearson.")))
    return f


def source_data():
    rng = np.random.default_rng(20261025); sd = sqrt(0.08)
    MA = [(0.4, 0.9), (2.0, 1.8), (2.3, 2.3), (2.6, 1.8)]; MB = [(1.5, 1.0), (1.9, 1.0), (1.5, 3.0), (3.3, 2.6)]
    pa = [(float(x), float(y)) for m in MA for x, y in rng.normal(m, sd, (50, 2))]
    pb = [(float(x), float(y)) for m in MB for x, y in rng.normal(m, sd, (50, 2))]
    spec = {
        "id": "lec05-kp11-data",
        "alt": tr("來源的模擬設定：A 類四群、B 類四群，每群 50 點，共 8×50=400 點，每群協方差 0.08I（標準差 √0.08≈0.283）；圖是依相同設定重抽的示意點。網路 2–3–2–1，logistic 活化，參數數 9+8+3=20；動量 μ=0.05、α=0.85；自適應動量 μ=0.01、α=0.85、r_i=1.05、c=1.05、r_d=0.7，來源報告批次模式下自適應較快。",
                  "The source's simulation setup: four clusters of class A and four of class B, 50 points each, 8×50=400 points, each with covariance 0.08I (standard deviation √0.08≈0.283); the figure is a redrawn sketch of the same setup. Network 2–3–2–1 with logistic activations and 9+8+3=20 parameters; momentum μ=0.05, α=0.85; adaptive momentum μ=0.01, α=0.85, r_i=1.05, c=1.05, r_d=0.7; the source reports that adaptive momentum is faster in batch mode."),
        "plot": {"x": [-0.4, 4.4], "y": [-0.4, 4.2], "x_ticks": [0, 1, 2, 3, 4], "y_ticks": [0, 1, 2, 3, 4], "x_label": tr("x₁（重抽的示意點）", "x₁ (redrawn sketch points)"), "y_label": "x₂", "w": 400, "h": 300, "legend_pos": "left"},
        "series": [{"kind": "dots", "points": pa, "role": "blue", "d": 7, "name": tr("A 類（4 群）", "class A (4 clusters)")},
                   {"kind": "dots", "points": pb, "role": "orange", "d": 7, "shape": "square", "name": tr("B 類（4 群）", "class B (4 clusters)")}],
        "side": [{"title": tr("資料", "Data"), "role": "blue", "h": 1.5, "text": tr("每群 50 點，$8\\times50=400$ 點；\n每群協方差 $0.08I$，\n標準差 $\\sqrt{0.08}\\approx0.283$。", "50 points per cluster, $8\\times50=400$;\ncovariance $0.08I$, std. dev. $\\sqrt{0.08}\\approx0.283$.")},
                 {"title": tr("網路", "Network"), "role": "green", "h": 1.0, "text": tr("2–3–2–1，logistic 活化，目標 1／0；\n含偏置的參數 $9+8+3=20$。", "2–3–2–1, logistic activation, targets 1/0;\nparameters with biases $9+8+3=20$.")},
                 {"title": tr("兩組動量設定", "Two momentum settings"), "role": "orange", "h": 1.7, "text": tr("動量：$\\mu=0.05$、$\\alpha=0.85$。\n自適應：$\\mu=0.01$、$\\alpha=0.85$、\n$r_i=1.05$、$c=1.05$、$r_d=0.7$。\n批次模式下自適應較快；此 $\\alpha$ 是動量，不是懲罰比重。", "momentum: $\\mu=0.05$, $\\alpha=0.85$.\nadaptive: $\\mu=0.01$, $\\alpha=0.85$,\n$r_i=1.05$, $c=1.05$, $r_d=0.7$.\nAdaptive is faster in batch mode; $\\alpha$ here is momentum.")}],
        "takeaway": tr("八群互相穿插，兩類不能用一條直線分開；以上設定都是來源的報告。", "The eight clusters interleave, so no single line separates the classes; the settings above are the source's report."),
    }
    return "plot_xy", spec


def pruning_report():
    spec = {
        "id": "lec05-kp11-counts",
        "alt": tr("來源的剪枝報告：兩個各 20 單元的隱藏層，先訓練含 480 個權重的模型，每 100 輪檢查顯著度，最後保留 25 個、刪除 455 個；25/480≈5.21%，455/480≈94.79%。來源報告效能改善、邊界由彎曲變得近似直線；單一案例，不保證剪枝普遍提升泛化。",
                  "The source's pruning report: two hidden layers of 20 units each, a model with 480 weights is trained first, the saliency is checked every 100 epochs, and finally 25 weights are kept and 455 deleted; 25/480≈5.21%, 455/480≈94.79%. The source reports better performance and a boundary that goes from curved to nearly straight; one case, no guarantee that pruning generally improves generalization."),
        "unit": tr("權重數（來源報告）", "number of weights (source report)"),
        "bars": [{"label": tr("剪枝前", "before pruning"), "value": 480, "role": "blue", "note": tr("起點", "start")},
                 {"label": tr("刪除", "deleted"), "value": 455, "role": "red", "note": "455/480 ≈ 94.79%"},
                 {"label": tr("保留", "kept"), "value": 25, "role": "green", "note": "25/480 ≈ 5.21%"}],
        "max": 700, "ticks": [0, 100, 200, 300, 400, 500, 600, 700],
        "takeaway": tr("來源報告效能改善、邊界由彎曲變近似直線；這是單一案例，不保證剪枝普遍提升泛化。", "The source reports better performance and a nearly straight boundary; one case, not a general guarantee."),
    }
    return "bars_compare", spec


def compensation():
    spec = {
        "id": "lec05-kp11-comp",
        "alt": tr("補充二次例子：H11=4、H12=1、H22=2，刪除 w1=0.5（δw1=−0.5）。δw2=0 時代價 ½×4×0.5²=0.5；允許補償，ΔJ=0.5−0.5δw2+δw2²；對 δw2 微分 −0.5+2δw2=0 得 δw2=0.25，代價 0.5−0.125+0.0625=0.4375，比 0.5 小。",
                  "Supplementary quadratic example: H11=4, H12=1, H22=2, delete w1=0.5 (δw1=−0.5). With δw2=0 the cost is ½×4×0.5²=0.5; allowing compensation, ΔJ=0.5−0.5δw2+δw2²; differentiating, −0.5+2δw2=0 gives δw2=0.25 and the cost 0.5−0.125+0.0625=0.4375, smaller than 0.5."),
        "steps": [{"role": "blue", "title": tr("① 只刪 w₁", "① Delete w₁ only"), "lines": ["$H_{11}=4,\\ \\delta w_1=-0.5$", "$\\delta w_2=0$", "$\\frac{1}{2}\\times4\\times0.5^2$"], "result": "$\\Delta J=0.5$"},
                  {"role": "green", "title": tr("② 允許 w₂ 補償", "② Let w₂ compensate"), "lines": ["$H_{12}=1,\\ H_{22}=2$", "$\\Delta J=0.5-0.5\\delta w_2+\\delta w_2^2$"], "result": tr("交叉項出現", "cross term appears")},
                  {"role": "orange", "title": tr("③ 取最小", "③ Take the minimum"), "lines": ["$-0.5+2\\delta w_2=0$", "$\\delta w_2=0.25$", "0.5 − 0.125 + 0.0625 = 0.4375"], "result": tr("比 0.5 小", "below 0.5")}],
        "takeaway": tr("補償後代價 0.4375 比 0.5 小：只用對角元會漏掉這個效果，要用非對角元 H₁₂。", "With compensation the cost 0.4375 is below 0.5: the diagonal alone misses this, the off-diagonal H₁₂ is needed."),
    }
    return "steps_flow", spec


def solution():
    spec = {
        "id": "lec05-kp11-solution",
        "alt": tr("解答：題 1：s_C=½×8×0.5²=1，s_D=½×0.1×2²=0.2，依估計先刪 D；題 2：ρ(2)=2²/(2²+2²)=4/8=0.5，權重仍是 2，只有執行設零或移除才叫刪除；驗算：較大的權重也可以有較小的顯著度。",
                  "Solution: Q1: s_C=½×8×0.5²=1, s_D=½×0.1×2²=0.2, so by the estimate remove D first; Q2: ρ(2)=2²/(2²+2²)=4/8=0.5, the weight is still 2 and only actually zeroing or removing it counts as deleting; check: a larger weight can have a smaller saliency."),
        "steps": [{"role": "blue", "title": tr("① 題 1：顯著度", "① Q1: saliency"), "lines": ["$s_C=\\frac{1}{2}\\times8\\times0.5^2=1$", "$s_D=\\frac{1}{2}\\times0.1\\times2^2=0.2$"], "result": tr("先刪 D", "remove D first")},
                  {"role": "green", "title": tr("② 題 2：懲罰值", "② Q2: penalty"), "lines": ["$\\rho(2)=2^2/(2^2+2^2)$", "$=4/8$"], "result": "$\\rho=0.5$"},
                  {"role": "orange", "title": tr("③ 它沒被刪", "③ Not deleted"), "lines": [tr("權重仍是 2", "the weight is still 2"), tr("設零或移除才叫刪除", "only zeroing or removing deletes")], "result": tr("懲罰 ≠ 刪除", "penalty ≠ deletion")}],
        "takeaway": tr("較大的權重也可以有較小的顯著度；懲罰值只是代價的一部分，不是刪除。", "A larger weight can have a smaller saliency; a penalty value is a cost term, not a deletion."),
    }
    return "steps_flow", spec
