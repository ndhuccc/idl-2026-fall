"""KP11 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp11_nums as K
W, H = 1160, 480

def arrow(x1, y1, x2, y2, col="#b7b0a5", w=3):
    a = math.atan2(y2 - y1, x2 - x1); L = 12
    p = [(x2, y2), (x2 - L * math.cos(a - .45), y2 - L * math.sin(a - .45)), (x2 - L * math.cos(a + .45), y2 - L * math.sin(a + .45))]
    return f'<line x1="{x1}" y1="{y1}" x2="{x2 - 8 * math.cos(a):.1f}" y2="{y2 - 8 * math.sin(a):.1f}" stroke="{col}" stroke-width="{w}"/><polygon points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in p)}" fill="{col}"/>'
def dline(ax, x0, y0, x1, y1, col, w=2, dash="7 6"): ax.raw(f'<line x1="{ax.X(x0):.1f}" y1="{ax.Y(y0):.1f}" x2="{ax.X(x1):.1f}" y2="{ax.Y(y1):.1f}" stroke="{col}" stroke-width="{w}" stroke-dasharray="{dash}"/>'); return ax
def xlab(ax, s): ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+52}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{s}</text>')

# ------------------------------------------------ 1: deletion amount vs deletion cost (curvature)
def page1(lang):
    t = tr(lang); o = ""
    specs = [(0, "blue", "A", .2, 100, (-.3, .3), [-.2, 0, .2], t("A：權重 0.2，曲線陡（曲率 H = 100）", "A: weight 0.2, steep curve (curvature H = 100)")),
             (404, "orange", "B", 1, .02, (-1.2, 1.2), [-1, 0, 1], t("B：權重 1，曲線平（曲率 H = 0.02）", "B: weight 1, flat curve (curvature H = 0.02)"))]
    for x0, role, nm, w, Hc, xr, xt, title in specs:
        o += cardH(x0, 0, 392, 404, role, title, 17)
        ax = Axes(x0 + 62, 62, 304, 236, xr, (0, 2.4)); ax.frame(xt, [0, 1, 2], xnames={v: f"{v:g}".replace("-", "−") for v in xt})
        xm = min(xr[1], math.sqrt(2 * 2.4 / Hc)); xs = np.linspace(-xm, xm, 50)
        ax.line([(float(a), K.sal(Hc, a)) for a in xs], hexc(role), 4)
        c = K.sal(Hc, w); dline(ax, -w, 0, -w, c, "#e0c85f"); ax.dot(-w, c, "#e0c85f", 8)
        xlab(ax, t("修改量 δw（刪除就是 δw = −w）", "change δw (deleting means δw = −w)"))
        o += ax.html()
        lbl = t(f"刪除代價 ΔJ = {c:g}", f"deletion cost ΔJ = {c:g}")
        o += chip(x0 + 130, 80 if nm == "A" else 190, 232, "yellow", lbl, True, 34, 18) if nm == "A" else chip(x0 + 130, 170, 232, "yellow", lbl, True, 34, 18)
        o += txt(x0 + 12, 354, 368, t(f"ΔJ = ½ × {Hc:g} × {w:g}² = {c:g}", f"ΔJ = ½ × {Hc:g} × {w:g}² = {c:g}"), 19, TXT, True, "center", 30)
    o += cardH(812, 0, 348, 404, "violet", t("名詞", "Terms"))
    o += named(826, 50, 320, t("剪枝：移除部分連線或單元，減少模型可調的部分；刪除後仍需檢查表現。", "Pruning: remove some connections or units so the model has fewer adjustable parts; check performance after deleting."), "term:" + t("剪枝", "Pruning"), 16, TXT, 100)
    o += named(826, 160, 320, t("曲率：損失的斜率改變得多快，用二階導數衡量；曲率大，走一小步代價就變很多。", "Curvature: how fast the slope of the loss changes, measured by the second derivative; with large curvature a small step changes the cost a lot."), "term:" + t("曲率", "Curvature"), 16, TXT, 124)
    o += named(826, 290, 320, t("海森矩陣：把所有二階偏導數排成矩陣 H；H_ii 是第 i 個方向的曲率。", "Hessian: the matrix H of all second partial derivatives; H_ii is the curvature along direction i."), "term:" + t("海森矩陣", "Hessian"), 16, TXT, 100)
    o += take(t, "權重小的 A 刪除代價反而是 B 的 200 倍：只改一點點，不等於影響一定小。", "Small-weight A costs 200 times as much to delete as B: changing a little does not mean affecting little.", 418)
    return o

# ------------------------------------------------ 2: saliency derivation
def page2(lang):
    t = tr(lang); o = ""
    sA, sB = K.sal(100, .2), K.sal(.02, 1)
    o += cardH(0, 0, 700, 404, "blue", t("顯著度的四步推導", "Deriving saliency in four steps"))
    rows = [(t("① 泰勒展開", "① Taylor expansion"), r"\Delta J\approx\sum_i g_i\delta w_i+\frac{1}{2}\sum_{i,j}H_{ij}\delta w_i\delta w_j"),
            (t("② 接近極小值 g≈0", "② g≈0 near a minimum"), r"\Delta J\approx\frac{1}{2}\sum_{i,j}H_{ij}\delta w_i\delta w_j"),
            (t("③ 對角近似", "③ drop cross terms"), r"\Delta J\approx\frac{1}{2}\sum_i H_{ii}(\delta w_i)^2"),
            (t("④ 代入 δwᵢ = −wᵢ", "④ put δwᵢ = −wᵢ"), r"s_i=\frac{1}{2}H_{ii}w_i^2")]
    for i, (lab, tex) in enumerate(rows):
        y0 = 50 + i * 88; hot = i == 3
        o += card(12, y0, 676, 78, "yellow" if hot else "grey", .2 if hot else .1)
        o += chip(22, y0 + 20, 200, "yellow" if hot else "blue", lab, hot, 38, 17) + formula(236, y0 + 8, 440, tex, 22 if not hot else 26, 62)
    o += cardH(716, 0, 444, 404, "green", t("A 與 B 的顯著度（對數刻度）", "Saliency of A and B (log scale)"))
    ax = Axes(790, 72, 330, 130, (-3, 1), (0, 1))
    def L(v): return 790 + (math.log10(v) + 3) / 4 * 330
    for v in (.001, .01, .1, 1, 10):
        o += svg(0, 0, W, H, f'<line x1="{L(v):.1f}" y1="64" x2="{L(v):.1f}" y2="216" stroke="#6b645a" stroke-width="1" opacity="0.5"/>') + txt(L(v) - 36, 218, 72, f"{v:g}", 16, MUT, False, "center", 24)
    for i, (nm, s, c1, c2) in enumerate((("A", sA, "#d9822b", "#a8452b"), ("B", sB, "#6a9fd6", "#2f4f7a"))):
        y0 = 78 + i * 62; o += box(790, y0, L(s) - 790, 44, c1, c2, 180 - 90, shadow=False, r=8) + txt(730, y0 + 6, 52, nm, 22, TXT, True, "center", 32) + txt(L(s) + 6, y0 + 8, 90, f"{s:g}", 21, TXT, True, "left", 30)
    o += chip(746, 250, 392, "yellow", t("2 ÷ 0.01 = 200 倍：依估計先刪 B", "2 ÷ 0.01 = 200 times: delete B first"), True, 40, 19)
    o += named(730, 304, 416, t("顯著度 sᵢ：刪掉第 i 個權重時代價增加量的估計；是近似，不是實測損失。", "Saliency sᵢ: an estimate of how much the cost rises when weight i is deleted; an approximation, not a measured loss."), "term:" + t("顯著度", "Saliency"), 16, TXT, 76)
    o += take(t, "A：½ × 100 × 0.2² = 2；B：½ × 0.02 × 1² = 0.01。權重較大的 B 反而先刪。", "A: ½ × 100 × 0.2² = 2; B: ½ × 0.02 × 1² = 0.01. The larger-weight B is deleted first.", 418)
    return o

# ------------------------------------------------ 3: conditions + procedure
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 1160, 214, "blue", t("可檢查的剪枝程序：四步循環", "A checkable pruning procedure: a four-step loop"))
    steps = [(t("先訓練到合適的位置", "Train to a good point"), "blue"), (t("估每條連線的顯著度", "Estimate each connection's saliency"), "teal"), (t("刪少量低顯著度的連線", "Delete a few low-saliency connections"), "orange"), (t("再訓練並檢查驗證表現", "Retrain and check on validation data"), "green")]
    cx = []
    for i, (s, role) in enumerate(steps):
        x0 = 22 + i * 280; cx.append(x0 + 125); o += card(x0, 52, 250, 92, role, .22) + chip(x0 + 10, 60, 40, role, str(i + 1), True, 34, 19) + txt(x0 + 54, 62, 190, s, 17, TXT, True, "left", 80)
        if i < 3: o += svg(0, 0, W, H, arrow(x0 + 252, 98, x0 + 278, 98, "#b7b0a5"))
    o += svg(0, 0, W, H, f'<line x1="{cx[3]}" y1="146" x2="{cx[3]}" y2="184" stroke="#e0c85f" stroke-width="3"/><line x1="{cx[3]}" y1="184" x2="{cx[1]}" y2="184" stroke="#e0c85f" stroke-width="3"/>' + arrow(cx[1], 184, cx[1], 148, "#e0c85f"))
    o += txt(cx[1] + 20, 160, 560, t("回到第 2 步：每刪一批都重新評估，不把一個排序永久當成真理", "Back to step 2: re-estimate after every batch; never treat one ranking as permanent truth"), 16, "#e0c85f", True, "left", 24)
    o += cardH(0, 226, 1160, 178, "orange", t("顯著度只是估計：這四種情況下會失準", "Saliency is only an estimate: it fails in these four cases"))
    cs = [(t("尚未接近極小值", "Not yet near a minimum"), t("一階項 g 未必小，第一項不能省。", "The first-order term g need not be small, so it cannot be dropped.")),
          (t("刪除幅度太大", "Deleting too much"), t("高階項也可能不可忽略。", "Higher-order terms may no longer be negligible.")),
          (t("同時刪多個權重", "Deleting several at once"), t("權重之間可能耦合，對角近似失準。", "Weights may be coupled, so the diagonal approximation breaks.")),
          (t("海森矩陣有負曲率", "Negative curvature"), t("負數不能直接讀成「安全」。", "A negative value must not be read as \"safe\"."))]
    for i, (a, b) in enumerate(cs):
        x0 = 14 + i * 285; o += card(x0, 270, 270, 124, "orange", .12) + txt(x0 + 8, 276, 254, a, 17, TXT, True, "center", 26) + txt(x0 + 8, 308, 254, b, 16, TXT, False, "center", 24)
    o += take(t, "所以：小步驟刪除、重新量測代價、再訓練剩餘權重，並保留驗證表現最好的模型。", "So: delete in small steps, re-measure the cost, retrain the rest and keep the best validation model.", 418)
    return o

# ------------------------------------------------ 4: regularization
def page4(lang):
    t = tr(lang); o = ""
    o += card(0, 0, 1160, 78, "violet", .14)
    o += formula(14, 12, 520, r"J_{\mathrm{total}}=J_{\mathrm{data}}+\alpha P(\mathbf{w})", 26, 54)
    o += named(548, 6, 600, t("正則化：在資料誤差之外，加上對模型的限制或偏好；P 是權重懲罰，α ≥ 0 控制比重。α 會隨誤差尺度（總和或平均）改變，不能跨設定照抄。", "Regularization: add a restriction or preference on the model beyond the data error; P is the weight penalty and α ≥ 0 sets its share. α depends on the error scale (sum or mean), so do not copy it across setups."), "term:" + t("正則化", "Regularization"), 16, TXT, 72)
    S = 340; base = 352
    for x0, a, role in ((0, .1, "green"), (590, 0, "red")):
        o += cardH(x0, 90, 570, 314, role, t(f"α = {a:g}：總代價 = 資料誤差 + α × 懲罰", f"α = {a:g}: total cost = data error + α × penalty"), 17)
        pick = None
        for j, (nm, d, P) in enumerate((("A", .10, 4), ("B", .20, 1))):
            bx = x0 + 50 + j * 150; hd = d * S; hp = a * P * S; tot = K.total(d, P, a)
            o += box(bx, base - hd, 110, hd, "#6a9fd6", "#2f4f7a", 180, shadow=False, r=6) + txt(bx, base - hd + hd / 2 - 13, 110, f"{d:.2f}", 16, TXT, True, "center", 26)
            if hp > 0: o += box(bx, base - hd - hp, 110, hp, "#d9822b", "#a8452b", 180, shadow=False, r=6) + txt(bx, base - hd - hp + hp / 2 - 13, 110, f"{a*P:.2f}", 16, TXT, True, "center", 26)
            o += txt(bx - 10, base - hd - hp - 32, 130, f"{tot:.2f}", 20, TXT, True, "center", 28) + txt(bx, base + 6, 110, t(f"模型 {nm}", f"model {nm}"), 17, TXT, True, "center", 26)
        o += svg(0, 0, W, H, f'<line x1="{x0+30}" y1="{base}" x2="{x0+340}" y2="{base}" stroke="#b7b0a5" stroke-width="2"/>')
        tA, tB = K.total(.1, 4, a), K.total(.2, 1, a); win = "B" if tB < tA else "A"
        o += txt(x0 + 352, 140, 212, t("A：", "A: ") + f"0.10 + {a:g} × 4 = {tA:.2f}", 17, TXT, False, "left", 26) + txt(x0 + 352, 176, 212, t("B：", "B: ") + f"0.20 + {a:g} × 1 = {tB:.2f}", 17, TXT, False, "left", 26)
        o += chip(x0 + 352, 230, 200, role, t(f"選 {win}", f"pick {win}"), True, 44, 22)
        o += txt(x0 + 352, 290, 212, t("藍：資料誤差；橘：α × 懲罰", "blue: data error; orange: α × penalty") if a else t("沒有懲罰，只看資料誤差", "no penalty: data error only"), 16, MUT, False, "left", 48)
    o += take(t, "α 要用驗證資料選；總代價較小，不等於測試誤差必然較小。", "Choose α on validation data; a smaller total cost does not guarantee a smaller test error.", 418)
    return o

# ------------------------------------------------ 5: saturating penalty
def page5(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 404, "green", t("飽和懲罰 ρ 與它的導數（橫軸為權重 w，τ = 1）", "The saturating penalty ρ and its derivative (horizontal axis: weight w, τ = 1)"))
    ax = Axes(86, 62, 590, 270, (-4, 4), (-.8, 1.1)); ax.frame([-4, -2, 0, 2, 4], [-.5, 0, .5, 1], xnames={v: f"{v:g}".replace("-", "−") for v in (-4, -2, 0, 2, 4)}, yfmt=lambda v: f"{v:g}".replace("-", "−"))
    dline(ax, -4, 1, 4, 1, "#b7b0a5", 2, "4 6")
    xs = np.linspace(-4, 4, 70)
    ax.line([(float(a), K.rho(a)) for a in xs], GR, 4).line([(float(a), K.drho(a)) for a in xs], OR, 4)
    for w in (.2, 1, 3):
        ax.dot(w, K.rho(w), "#ffffff", 8).dot(w, K.rho(w), GR, 5)
    ax.text(.2, K.rho(.2), "0.0385", TXT, "start", 17, True, dx=10, dy=30); ax.text(1, K.rho(1), "0.5", TXT, "start", 17, True, dx=14, dy=34); ax.text(3, K.rho(3), "0.9", TXT, "start", 17, True, dx=-6, dy=32)
    ax.text(.15, -.5, "ρ′(0) = 0", OR, "start", 17, True)
    o += ax.html()
    o += legend([(GR, t("ρ(w)：懲罰", "ρ(w): penalty")), (OR, t("ρ′(w)：導數", "ρ′(w): derivative"))], 470, 252, 200)
    o += txt(14, 372, 672, t("w = 0.2、1、3 時 ρ ≈ 0.0385、0.5、0.9：小權重幾乎不罰，大權重飽和到 1。〔程式計算〕", "w = 0.2, 1, 3: ρ ≈ 0.0385, 0.5, 0.9; large weights saturate at 1. (computed)"), 16, TXT, False, "center", 24)
    o += cardH(716, 0, 444, 404, "violet", t("公式與名詞", "Formulas and terms"))
    o += formula(724, 46, 428, r"\rho(w)=\frac{w^2}{\tau^2+w^2},\quad P=\sum_i\rho(w_i)", 21, 56)
    o += formula(724, 106, 428, r"\rho'(w)=\frac{2\tau^2 w}{(\tau^2+w^2)^2}", 21, 56)
    o += named(730, 172, 416, t("飽和懲罰：τ > 0 是尺度；權重很大時懲罰接近 1，叫飽和。比較尺度要用絕對值：|w| ≪ τ 才算接近零。", "Saturating penalty: τ > 0 is the scale; for very large weights the penalty approaches 1, called saturation. Compare scales with absolute values: |w| ≪ τ means near zero."), "term:" + t("飽和懲罰", "Saturating penalty"), 16, TXT, 124)
    o += named(730, 304, 416, t("權重消除：訓練時促進縮小，再把 |wᵢ| < ε 的權重設零或刪除；ε 是刪除門檻，τ 是懲罰尺度。", "Weight elimination: encourage shrinking during training, then zero or delete weights with |wᵢ| < ε; ε is the deletion threshold and τ the penalty scale."), "term:" + t("權重消除", "Weight elimination"), 16, TXT, 96)
    o += take(t, "導數在 w = 0 為零：有限步更新不保證權重恰好到零，所以要另設門檻 ε 才真正刪除。", "The derivative is zero at w = 0: finite updates do not guarantee exactly zero, so a threshold ε does the deleting.", 418)
    return o

# ------------------------------------------------ 6: operations and norms
def page6(lang):
    t = tr(lang); o = ""
    ops = [("blue", t("刪連線", "Delete a connection"), t("把一個權重設成 0，單元還在，其他連線不變。", "Set one weight to 0; the unit stays and the other connections are unchanged."), t("檢查：刪除後的資料表現。", "Check: performance on the data after deleting.")),
           ("orange", t("刪單元", "Delete a unit"), t("整個單元連同它所有的連線一起移除。", "Remove the whole unit together with all its connections."), t("檢查：下一層是否少了這條訊號路徑。", "Check: whether the next layer lost this signal path.")),
           ("green", t("縮小數值", "Shrink values"), t("權重或單元輸出變小，連線與單元都還在。來源也對神經元輸出加懲罰。", "Weights or unit outputs get smaller; connections and units remain. The source also penalizes neuron outputs."), t("檢查：輸出小仍可能被很大的下一層權重放大，要看整條路徑。", "Check: a small output can still be amplified by large next-layer weights; look at the whole path."))]
    for i, (role, name, a, b) in enumerate(ops):
        x0 = i * 276; o += cardH(x0, 0, 262, 404, role, name, 18)
        o += chip(x0 + 12, 52, 238, role, t("做了什麼", "What changes"), False, 30, 16) + txt(x0 + 12, 90, 238, a, 17, TXT, False, "left", 24)
        o += chip(x0 + 12, 232, 238, "yellow", t("要檢查什麼", "What to check"), False, 30, 16) + txt(x0 + 12, 270, 238, b, 17, TXT, False, "left", 24)
    o += cardH(836, 0, 324, 404, "violet", t("來源的複雜度項", "The source's complexity term"))
    o += formula(844, 44, 308, r"A^3\sqrt{\log d/N}", 26, 52)
    o += named(850, 100, 296, t("A：每個單元權重絕對值和的共同上界；d：輸入維度；N：樣本數。", "A: a common bound on each unit's sum of absolute weights; d: input dimension; N: sample size."), "term:" + t("範數上界 A", "Norm bound A"), 16, TXT, 100)
    base = 332
    for j, (A, v) in enumerate(((1, 1), (2, 8))):
        h_ = v * 11; bx = 890 + j * 120
        o += box(bx, base - h_, 80, h_, "#a98bd0" if j == 0 else "#d97757", "#5e4a85" if j == 0 else "#94503a", 180, shadow=False, r=6) + txt(bx - 10, base - h_ - 32, 100, f"×{v}", 22, TXT, True, "center", 30) + txt(bx - 10, base + 4, 100, f"A = {A}", 17, TXT, True, "center", 26)
    o += txt(852, 366, 292, t("固定 d 與 N，A 加倍：2³ = 8 倍", "Fix d, N; double A: 2³ = 8 times"), 17, TXT, True, "center", 28)
    o += take(t, "刪連線、刪單元、縮小數值要分別記錄；A 的三次方只是量級摘要，不是分類錯誤率。", "Record the three operations separately; A cubed is an order of magnitude, not an error rate.", 418)
    return o

# ------------------------------------------------ 7: cascade correlation
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 236, "blue", t("級聯相關法：兩個階段不能混成同一個目標", "Cascade correlation: two stages that must not be mixed into one objective"), 17)
    o += chip(24, 48, 316, "blue", t("階段一：找候選單元", "Stage 1: find a candidate unit"), True, 30, 17) + chip(358, 48, 318, "green", t("階段二：重訓輸出側", "Stage 2: retrain the output side"), True, 30, 17)
    steps = [("blue", t("從小架構出發", "Start from a small architecture")), ("blue", t("訓練候選單元的輸入側權重，使 S 最大", "Train the candidate's input weights to maximize S")), ("green", t("凍結這些輸入側權重", "Freeze these input weights")), ("green", t("重訓輸出側，降低平方誤差", "Retrain the output side to lower the squared error"))]
    for i, (role, s) in enumerate(steps):
        x0 = 24 + i * 169; o += card(x0, 90, 154, 130, role, .2) + chip(x0 + 8, 98, 32, role, str(i + 1), True, 28, 17) + txt(x0 + 4, 130, 146, s, 16, TXT, True, "center", 24)
    o += cardH(716, 0, 444, 236, "violet", t("名詞", "Terms"))
    o += named(728, 50, 420, t("殘差：真實目標減去目前預測，eₙ = tₙ − ŷₙ，n 是樣本編號。", "Residual: the true target minus the current prediction, eₙ = tₙ − ŷₙ, n is the sample index."), "term:" + t("殘差", "Residual"), 16, TXT, 76)
    o += named(728, 134, 420, t("級聯相關法：每次增加一個能回應目前殘差的隱藏單元，再重訓輸出。新單元接收原輸入與先前的隱藏單元。", "Cascade correlation: add one hidden unit at a time that responds to the current residual, then retrain the output; the new unit receives the inputs and earlier hidden units."), "term:" + t("級聯相關法", "Cascade correlation"), 16, TXT, 100)
    o += cardH(0, 248, 1160, 156, "yellow", t("教學分數 S（單輸出）：候選輸出 v 與殘差 e", "Teaching score S (single output): candidate output v against residual e"), 17)
    o += formula(14, 292, 520, r"S=\left|\sum_n(v_n-\bar{v})(e_n-\bar{e})\right|", 24, 62)
    o += txt(14, 360, 520, t("橫線是樣本平均；沒有除標準差，不是 Pearson 係數。", "The bar is the sample mean; no division by the standard deviation, so it is not a Pearson coefficient."), 16, MUT, False, "center", 48)
    for i, (role, hd, tex, verdict) in enumerate((("green", t("候選 v = (−1, 1)", "candidate v = (−1, 1)"), r"S=|(-1)(-1)+1\times 1|=2", t("S = 2：與殘差同步", "S = 2: moves with the residual")),
                                               ("red", t("常數候選 v = (1, 1)", "constant candidate v = (1, 1)"), r"v-\bar{v}=(0,0),\ S=0", t("S = 0：完全沒回應", "S = 0: no response")))):
        x0 = 560 + i * 296; o += card(x0, 290, 284, 104, role, .16) + txt(x0, 294, 284, hd, 16, TXT, True, "center", 24) + formula(x0, 320, 284, tex, 18, 40) + txt(x0, 364, 284, verdict, 16, TXT, True, "center", 24)
    o += take(t, "取 e = (−1, 1)：選 S 最大的候選，凍結它的輸入側權重，再去重訓輸出。", "With e = (−1, 1): pick the candidate with the largest S, freeze its input weights, then retrain the output.", 418)
    return o

# ------------------------------------------------ 8: source data and setup
def page8(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 404, "blue", t("8 群各 50 點：400 個重抽樣本", "8 clusters of 50 points: 400 redrawn samples"), 17)
    ax = Axes(50, 58, 490, 262, (-.5, 4.5), (-.2, 4.2)); ax.frame([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    for lab, X in K.data():
        for a, b in X: ax.dot(float(a), float(b), BL if lab == 0 else OR, 4)
    o += ax.html()
    o += legend([(BL, t("A 類", "class A")), (OR, t("B 類", "class B"))], 66, 64, 110)
    o += txt(10, 368, 540, t("種子 20261025 重抽：只核對分布，非原始樣本，未重跑訓練。", "Redrawn, seed 20261025: distribution check only."), 16, MUT, False, "center", 24)
    o += cardH(576, 0, 584, 404, "orange", t("來源的資料、網路與動量設定", "The source's data, network and momentum settings"), 17)
    o += card(588, 46, 560, 110, "blue", .12) + txt(596, 50, 544, t("資料：每群協方差 0.08 I（I 是二維單位矩陣）", "Data: each cluster has covariance 0.08 I (I is the 2-D identity)"), 16, TXT, True, "left", 24) + txt(596, 76, 544, t("8 × 50 = 400 點；標準差 √0.08 ≈ 0.283（不是 0.08）", "8 × 50 = 400 points; standard deviation √0.08 ≈ 0.283 (not 0.08)"), 17, TXT, False, "left", 26)
    o += txt(596, 106, 544, t("A 均值 (0.4,0.9) (2.0,1.8) (2.3,2.3) (2.6,1.8)<br>B 均值 (1.5,1.0) (1.9,1.0) (1.5,3.0) (3.3,2.6)", "A means (0.4,0.9) (2.0,1.8) (2.3,2.3) (2.6,1.8)<br>B means (1.5,1.0) (1.9,1.0) (1.5,3.0) (3.3,2.6)"), 16, MUT, False, "left", 48)
    o += card(588, 166, 560, 78, "teal", .12) + txt(596, 170, 544, t("網路 2–3–2–1，logistic 活化，目標 1／0", "Network 2–3–2–1, logistic activations, targets 1/0"), 17, TXT, True, "left", 26) + txt(596, 200, 544, t("含偏置的參數數：9 + 8 + 3 = 20", "Parameters including biases: 9 + 8 + 3 = 20"), 17, TXT, False, "left", 26)
    o += card(588, 254, 560, 140, "violet", .12) + txt(596, 258, 544, t("動量：μ = 0.05，α = 0.85", "Momentum: μ = 0.05, α = 0.85"), 17, TXT, True, "left", 26) + txt(596, 286, 544, t("自適應動量：μ = 0.01，α = 0.85，<br>rᵢ = 1.05，c = 1.05，r_d = 0.7", "Adaptive momentum: μ = 0.01, α = 0.85,<br>rᵢ = 1.05, c = 1.05, r_d = 0.7"), 17, TXT, True, "left", 26)
    o += txt(596, 338, 544, t("μ：學習率；α：動量係數；rᵢ、r_d：放大與縮小倍率；c：觸發縮小的代價比門檻。批次模式下自適應動量較快（來源）。", "μ: rate; α: momentum; rᵢ, r_d: raise/lower factors; c: cost-ratio threshold. Faster in batch mode (source)."), 16, MUT, False, "left", 24)
    o += take(t, "這些都是來源的設定與報告；這裡的重抽散點只核對分布。", "These are the source's settings and reports; the redrawn scatter here only checks the distribution.", 418)
    return o

# ------------------------------------------------ 9: pruning report + supplementary contour
def clip(poly, a, b, c):   # keep a*x+b*y+c >= 0
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]; fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0: out.append(p)
        if (fp >= 0) != (fq >= 0): u = fp / (fp - fq); out.append((p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])))
    return out
def page9(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 404, "red", t("來源的剪枝報告：480 → 25 個權重", "The source's pruning report: 480 → 25 weights"), 17)
    base = 322; sc = 230 / 480
    for j, (v, lab, c1, c2) in enumerate(((480, t("剪枝前", "before"), "#6a9fd6", "#2f4f7a"), (455, t("刪除", "deleted"), "#d97757", "#94503a"), (25, t("保留", "kept"), "#7dbb8c", "#3f7a52"))):
        bx = 40 + j * 170; hh = v * sc
        o += box(bx, base - hh, 120, hh, c1, c2, 180, shadow=False, r=8) + txt(bx - 10, base - hh - 36, 140, str(v), 24, TXT, True, "center", 32) + txt(bx - 10, base + 6, 140, lab, 17, TXT, True, "center", 26)
    o += svg(0, 0, W, H, f'<line x1="26" y1="{base}" x2="540" y2="{base}" stroke="#b7b0a5" stroke-width="2"/>')
    o += chip(336, 120, 214, "green", t("保留 25 ÷ 480 ≈ 5.21%", "kept 25/480 ≈ 5.21%"), False, 34, 17) + chip(336, 164, 214, "red", t("刪除 455 ÷ 480 ≈ 94.79%", "deleted 455/480 ≈ 94.79%"), False, 34, 16)
    o += txt(12, 360, 536, t("每 100 輪檢查顯著度；兩個各 20 單元的隱藏層。", "Saliency checked every 100 epochs; two hidden layers of 20 units."), 16, MUT, False, "center", 24)
    o += cardH(576, 0, 584, 404, "teal", t("［補充］p = 0.5 的等高線（不是訓練實驗）", "[Supplement] The p = 0.5 contour (not a training experiment)"), 17)
    ax = Axes(626, 54, 460, 262, (0, 4), (0, 4)); box_ = [(0, 0), (4, 0), (4, 4), (0, 4)]
    edges = [-99, -3, -2, -1, 0, 1, 2, 3, 99]; s = ""
    for lo, hi in zip(edges, edges[1:]):
        poly = clip(clip(box_, -0.6, 1, -1 - lo), 0.6, -1, 1 + hi)   # lo <= x2-0.6x1-1 <= hi
        if len(poly) < 3: continue
        mid = max(min((lo + hi) / 2 if abs(lo) < 90 and abs(hi) < 90 else (hi - 1 if lo == -99 else lo + 1) if True else 0, 5), -5); p = float(K.logistic(mid))
        r_, g_, b_ = (int(217 + (106 - 217) * p), int(119 + (159 - 119) * p), int(87 + (214 - 87) * p))
        s += f'<polygon points="{" ".join(f"{ax.X(x):.1f},{ax.Y(y):.1f}" for x, y in poly)}" fill="rgb({r_},{g_},{b_})" fill-opacity="0.55"/>'
    ax.raw(s); ax.frame([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]); ax.line([(0, 1), (4, 3.4)], "#e0c85f", 5)
    o += ax.html()
    o += txt(900, 70, 190, "p > 0.5", 17, "#9cc3ee", True, "center", 26) + txt(840, 270, 190, "p < 0.5", 17, "#f0a58a", True, "center", 26)
    o += chip(660, 170, 262, "yellow", t("p = 0.5：x₂ = 0.6x₁ + 1", "p = 0.5: x2 = 0.6x1 + 1"), True, 32, 16)
    o += txt(590, 352, 560, t("r = x₂ − 0.6x₁ − 1，p = 1/(1 + e⁻ʳ)；401 × 401 網格，門檻 0.5，範圍 [0, 4]²。〔程式計算〕", "r = x2 − 0.6x1 − 1, p = 1/(1 + e⁻ʳ); 401 × 401 grid, threshold 0.5, range [0, 4]². (computed)"), 16, MUT, False, "center", 48)
    o += card(0, 414, 1160, 66, "yellow", .14) + txt(14, 420, 1132, t("來源只報告「效能改善、邊界由彎曲變近似直線」，不保證剪枝普遍提升泛化；480 的計數口徑來源沒交代，不反推架構；原實驗沒有重跑。", "The source reports better performance and a boundary going from curved to nearly straight; pruning need not always generalize better; how the 480 was counted is not stated, so no architecture is inferred; nothing was rerun."), 16, TXT, True, "center", 48)
    return o

# ------------------------------------------------ 10: second-order compensation + finite difference
def page10(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 404, "blue", t("允許 w₂ 補償：H = [4 1; 1 2]，刪除 w₁ = 0.5", "Allowing w₂ to compensate: H = [4 1; 1 2], delete w₁ = 0.5"), 17)
    o += formula(8, 44, 624, r"H_{11}=4,\ H_{12}=H_{21}=1,\ H_{22}=2;\quad \delta w_1=-0.5", 20, 40)
    rows = [("blue", t("① 只刪 w₁（δw₂ = 0）", "① delete w₁ only (δw₂ = 0)"), r"\Delta J=\frac{1}{2}\times4\times0.5^2=0.5"),
            ("orange", t("② 允許補償", "② allow compensation"), r"\Delta J=0.5-0.5\,\delta w_2+(\delta w_2)^2"),
            ("green", t("③ 對 δw₂ 微分 = 0", "③ set the derivative in δw₂ to 0"), r"-0.5+2\delta w_2=0\Rightarrow \delta w_2=0.25")]
    for i, (role, lab, tex) in enumerate(rows):
        y0 = 94 + i * 88; o += card(12, y0, 616, 78, role, .16) + chip(20, y0 + 8, 300, role, lab, False, 30, 16) + formula(20, y0 + 40, 600, tex, 20, 34)
    o += chip(12, 360, 616, "yellow", t("ΔJ = 0.5 − 0.125 + 0.0625 = 0.4375，比 0.5 小", "ΔJ = 0.5 − 0.125 + 0.0625 = 0.4375, smaller than 0.5"), True, 32, 17)
    o += cardH(656, 0, 504, 250, "orange", t("ΔJ 對 δw₂", "ΔJ against δw₂"), 17)
    ax = Axes(716, 54, 420, 140, (-.1, .6), (.4, .6)); ax.frame([0, .25, .5], [.4, .5, .6])
    ax.line([(float(a), K.comp(float(a)) ) for a in np.linspace(-.1, .6, 50)], OR, 4); ax.dot(0, .5, "#ffffff", 8).dot(0, .5, BL, 5); ax.dot(.25, K.comp(.25), "#ffffff", 8).dot(.25, K.comp(.25), "#e0c85f", 5)
    ax.text(0, .5, "0.5", TXT, "start", 17, True, dx=10, dy=-10); ax.text(.25, K.comp(.25), "0.4375", "#e0c85f", "middle", 17, True, dy=-18)
    xlab(ax, "δw₂"); o += ax.html()
    o += cardH(656, 262, 504, 142, "teal", t("不求二階導數時：中心差分", "Without second derivatives: central difference"), 17)
    o += formula(664, 304, 488, r"H_{ii}\approx\frac{J(w+h)-2J(w)+J(w-h)}{h^2}", 18, 50)
    o += txt(664, 356, 488, t("例：J = (w − 0.5)²，w = 0.5，h = 0.1：(0.01 − 0 + 0.01) ÷ 0.01 = 2，與解析值一致。h 是步長，不是刪除門檻。", "Example: J = (w − 0.5)², w = 0.5, h = 0.1: (0.01 − 0 + 0.01) ÷ 0.01 = 2. h is a step size, not a deletion threshold."), 16, TXT, False, "center", 48)
    o += take(t, "刪一個、其餘固定時交叉項消失；讓其他權重補償，就要重新考慮非對角元。", "Deleting one weight with the rest fixed drops the cross term; letting others compensate restores it.", 418)
    return o

# ------------------------------------------------ 11: exercises
def page11(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("C：權重 0.5、曲率 8；D：權重 2、曲率 0.1。依顯著度先刪誰？", "C: weight 0.5, curvature 8; D: weight 2, curvature 0.1. By saliency, which is deleted first?"), [r"s_C=\frac{1}{2}\times8\times0.5^2=1", r"s_D=\frac{1}{2}\times0.1\times2^2=0.2"], t("先刪 D（0.2 < 1）。", "Delete D first (0.2 < 1)."), t("核對：較大的權重也可以有較小的顯著度。", "Check: a larger weight can have a smaller saliency.")),
          ("red", t("題 2", "Q2"), t("取 τ = 2、w = 2，算 ρ(w)。它是否已經代表這個權重被刪掉？", "With τ = 2 and w = 2, compute ρ(w). Does it mean the weight was deleted?"), [r"\rho(2)=\frac{2^2}{2^2+2^2}", r"=\frac{4}{8}=0.5"], t("ρ = 0.5；權重仍是 2，沒有刪除。", "ρ = 0.5; the weight is still 2, not deleted."), t("核對：懲罰是目標的一部分，設零或移除才叫刪除。", "Check: the penalty is part of the objective; only zeroing or removing deletes."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 52, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 194, 538, 150, "grey", .1) + formula(x + 16, 200, 538, tex[0], 22, 66) + formula(x + 16, 274, 538, tex[1], 22, 66)
        o += chip(x + 16, 364, 538, role, ans, True, 42, 20)
        o += txt(x + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 12)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp11_" + fid, fn, W, H, out)
