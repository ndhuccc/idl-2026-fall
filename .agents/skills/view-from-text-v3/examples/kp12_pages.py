"""KP12 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp12_nums as K
W, H = 1160, 480
ON, OFF, EDGE = "#d9822b", "#2a2723", "#6b645a"

def grid(x0, y0, cell, mat, on=ON, fmt_=None, hi=None, hi_col="#e0c85f"):
    """svg grid; mat entries 0 are empty, nonzero filled; fmt_ prints numbers; hi=(i,j,h,w) highlights a window."""
    s = ""
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            fill = on if v else OFF
            s += f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell-3}" height="{cell-3}" rx="5" fill="{fill}" stroke="{EDGE}" stroke-width="1.5"/>'
            if fmt_ is not None and (v or fmt_ == "all"): s += f'<text x="{x0+j*cell+(cell-3)/2}" y="{y0+i*cell+cell/2+7}" text-anchor="middle" font-size="20" font-weight="700" fill="#f2ede4">{v}</text>'
    if hi:
        i, j, h, w = hi; s += f'<rect x="{x0+j*cell-2}" y="{y0+i*cell-2}" width="{w*cell+1}" height="{h*cell+1}" rx="6" fill="none" stroke="{hi_col}" stroke-width="5"/>'
    return svg(0, 0, W, H, s)

# ------------------------------------------------ 1: local sharing rule
def page1(lang):
    t = tr(lang); o = ""; x = K.image(1); tx = K.image(3)
    o += cardH(0, 0, 780, 480, "blue", t("6×6 影像：一條三格筆畫，向右平移兩欄", "A 6×6 image: a three-cell stroke shifted two columns right"))
    o += grid(40, 80, 50, x, hi=(1, 0, 3, 3)) + grid(440, 80, 50, tx)
    o += txt(40, 392, 300, "x", 18, TXT, True, "center", 28) + txt(440, 392, 300, "Tx", 18, TXT, True, "center", 28)
    o += svg(0, 0, W, H, f'<line x1="352" y1="230" x2="424" y2="230" stroke="#b7b0a5" stroke-width="4"/><polygon points="436,230 420,221 420,239" fill="#b7b0a5"/>')
    o += txt(330, 190, 120, t("向右 2 欄", "2 columns right"), 16, MUT, False, "center", 24)
    o += txt(30, 420, 720, t("黃框：一個 3×3 視窗（感受野）。希望同一條筆畫移動後仍屬同類，不是要每個數字都不變。", "Yellow frame: one 3×3 window (receptive field). The goal is that the same stroke stays in its class after moving, not that every number stays the same."), 16, TXT, False, "left", 52)
    o += cardH(796, 0, 364, 480, "orange", t("共享核與視窗分數", "Shared kernel and window score"))
    o += grid(836, 62, 46, K.K, on="#4f86bf") + txt(836, 206, 140, "K", 18, TXT, True, "center", 28)
    o += txt(990, 100, 160, t("視窗分數", "window score"), 16, MUT, True, "center", 24) + card(990, 130, 160, 56, "yellow", .2) + txt(990, 140, 160, "1 + 1 + 1 = 3", 19, TXT, True, "center", 30)
    o += named(812, 244, 332, t("權重共享：多條對應連線被限制使用同一個可調數值，更新時也只更新這個共同數值。", "Weight sharing: corresponding connections are forced to use one adjustable value, and updates change only that shared value."), "term:" + t("權重共享", "Weight sharing"), 16, TXT, 100)
    o += named(812, 356, 332, t("轉換不變性：對指定的轉換 T，分類結果滿足 c(Tx) = c(x)。要先說清楚是哪一種轉換。", "Transformation invariance: for a given transformation T the class satisfies c(Tx) = c(x); say which transformation first."), "term:" + t("轉換不變性", "Transformation invariance"), 16, TXT, 100)
    return o

# ------------------------------------------------ 2: feature maps
def page2(lang):
    t = tr(lang); o = ""; x = K.image(1); tx = K.image(3); f1 = K.fmap(x); f2 = K.fmap(tx)
    o += cardH(0, 0, 380, 480, "blue", t("F(x)：4×4 特徵圖", "F(x): the 4×4 feature map"))
    o += grid(70, 90, 64, f1, fmt_="all") + txt(20, 372, 340, t("亮處在第 0 欄：2、3、2、1", "Bright in column 0: 2, 3, 2, 1"), 17, TXT, True, "center", 28)
    o += cardH(396, 0, 380, 480, "teal", t("F(Tx)：亮處移到第 2 欄", "F(Tx): the bright part moves to column 2"))
    o += grid(466, 90, 64, f2, fmt_="all") + txt(416, 372, 340, t("同樣的 2、3、2、1，在第 2 欄", "The same 2, 3, 2, 1, now in column 2"), 17, TXT, True, "center", 28)
    o += cardH(792, 0, 368, 480, "orange", t("等變，然後不變", "Equivariant first, then invariant"))
    o += named(808, 48, 336, t("特徵圖：把每個視窗的分數依位置排成陣列，記為 F(x)；本例 4×4。", "Feature map: the window scores arranged by position, written F(x); 4×4 here."), "term:" + t("特徵圖", "Feature map"), 16, TXT, 76)
    o += named(808, 132, 336, t("平移等變性：輸入平移，輸出也對應平移，F(Tx) = TF(x)。", "Translation equivariance: shifting the input shifts the output correspondingly, F(Tx) = TF(x)."), "term:" + t("平移等變性", "Translation equivariance"), 16, TXT, 76)
    o += named(808, 216, 336, t("最大值彙整：整張特徵圖只取最大值 z = max F(x)。", "Global max pooling: keep only the maximum over the whole map, z = max F(x)."), "term:" + t("最大值彙整", "Global max pooling"), 16, TXT, 76)
    o += card(808, 306, 336, 70, "yellow", .16) + txt(816, 314, 320, f"z(x) = {f1.max()}　z(Tx) = {f2.max()}", 22, TXT, True, "center", 36) + txt(816, 346, 320, t("兩個都是 3：不變", "both 3: invariant"), 16, MUT, False, "center", 24)
    o += txt(808, 388, 336, t("限制：整數平移、步長、補邊要相容；彙整會丟掉位置。〔程式計算〕", "Limits: integer shifts, stride and padding must agree; pooling discards position. (computed)"), 16, MUT, False, "left", 76)
    return o

# ------------------------------------------------ 3: parameter count
def page3(lang):
    t = tr(lang); o = ""
    n, pos, nos, sh = K.params(6)
    o += cardH(0, 0, 560, 480, "red", t("不共享：每個位置自己的 10 個參數", "Not shared: 10 parameters per position"))
    s = ""
    for i in range(4):
        for j in range(4): s += f'<rect x="{60+j*118}" y="{74+i*72}" width="108" height="62" rx="8" fill="#c4572e" opacity="0.75" stroke="{EDGE}" stroke-width="1.5"/><text x="{60+j*118+54}" y="{74+i*72+40}" text-anchor="middle" font-size="22" font-weight="700" fill="#f2ede4">10</text>'
    o += svg(0, 0, W, H, s) + card(40, 372, 480, 80, "red", .2) + formula(40, 384, 480, rf"16\times(9+1)={nos}", 28, 56)
    o += cardH(576, 0, 584, 480, "green", t("共享：全部位置共用同一組", "Shared: one set for every position"))
    s = ""
    for i in range(4):
        for j in range(4): s += f'<rect x="{616+j*60}" y="{74+i*62}" width="52" height="54" rx="8" fill="#4f9a68" opacity="0.8" stroke="{EDGE}" stroke-width="1.5"/>'
    o += svg(0, 0, W, H, s) + box(884, 150, 240, 96, "#4f86bf", "#2f4f7a", 135, shadow=True) + txt(884, 168, 240, t("共享的一組", "one shared set"), 17, TXT, True, "center", 28) + txt(884, 200, 240, t("9 個權重 + 1 個偏置", "9 weights + 1 bias"), 16, TXT, False, "center", 26)
    o += card(592, 372, 552, 80, "green", .2) + formula(592, 384, 552, rf"9+1={sh}", 28, 56)
    o += txt(592, 316, 552, t("十六個位置仍然都要算一次；省下的是「可獨立調整的數字」。〔程式計算〕", "All sixteen positions are still computed; what is saved is the number of independently adjustable values. (computed)"), 16, TXT, False, "center", 36)
    return o

# ------------------------------------------------ 4: shared gradient
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 720, 480, "violet", t("共享權重 k 怎麼更新", "How the shared weight k is updated"))
    o += txt(16, 48, 688, t("k 同時進入許多位置的分數 r_p：把所有路徑的貢獻加起來（鏈鎖法則）。", "k enters the score r_p of many positions: add up the contributions of all paths (chain rule)."), 16, TXT, False, "left", 52)
    o += card(16, 104, 688, 76, "violet", .22) + formula(16, 114, 688, r"\frac{\partial J}{\partial k}=\sum_p\delta_p\,x_p", 28, 60)
    o += txt(16, 192, 688, t("其中 δ_p = ∂J/∂r_p 是位置 p 的分數對代價的偏導，x_p 是乘上 k 的輸入。", "Here δ_p = ∂J/∂r_p is the derivative of the cost with respect to the score at position p and x_p is the input multiplied by k."), 16, MUT, False, "left", 52)
    for i, (d, xv) in enumerate(((0.2, 3), (-0.1, 1))):
        y0 = 254 + i * 90; o += card(16, y0, 688, 80, "grey", .12) + txt(28, y0 + 8, 200, t(f"位置 {i+1}", f"position {i+1}"), 17, MUT, True, "left", 28) + formula(228, y0 + 6, 460, rf"\delta={d},\ x={xv}\ \Rightarrow\ {d}\times{xv}={d*xv:.1f}", 22, 60)
    o += cardH(736, 0, 424, 480, "orange", t("一步一步算", "Step by step"))
    g = 0.2 * 3 + (-0.1) * 1; k2 = 1 - 0.1 * g
    o += card(752, 52, 392, 100, "grey", .12) + formula(752, 62, 392, rf"\frac{{\partial J}}{{\partial k}}=0.6-0.1={g:.1f}", 24, 76)
    o += card(752, 168, 392, 100, "grey", .12) + formula(752, 178, 392, rf"k=1-0.1\times{g:.1f}={k2:.2f}", 24, 76)
    o += card(752, 284, 392, 88, "yellow", .16) + txt(760, 294, 376, t("η = 0.1、原 k = 1：更新後所有位置都用 0.95", "η = 0.1, old k = 1: afterwards every position uses 0.95"), 17, TXT, True, "center", 70)
    o += txt(752, 386, 392, t("活化函數的導數已含在 δ 之內，不要漏掉或重算。〔程式計算〕", "The derivative of any activation is already inside δ; do not drop or repeat it. (computed)"), 16, MUT, False, "left", 76)
    return o

# ------------------------------------------------ 5: three routes
def page5(lang):
    t = tr(lang); o = ""
    cols = [("blue", t("選不變特徵", "Choose invariant features"), t("先把資料轉成對指定變換不敏感的表示", "First turn the data into a representation that ignores the given transformation"), t("可能丟掉辨識所需的資訊", "It may discard information needed for recognition")),
            ("green", t("架構約束", "Architectural constraints"), t("共享權重加上適當彙整，明確限制哪些計算必須相同", "Shared weights plus a suitable pooling force certain computations to be identical"), t("只對相容的變換成立", "It holds only for compatible transformations")),
            ("orange", t("資料擴增", "Data augmentation"), t("用合理的變換產生更多訓練例子", "Generate more training examples with reasonable transformations"), t("有限例子通常不保證所有變換下都不變", "Finite examples usually do not guarantee invariance under every transformation"))]
    for i, (role, name, how, risk) in enumerate(cols):
        x = i * 390; o += card(x, 0, 380, 396, role) + head(x, 0, 380, role, name, 40, 19)
        o += txt(x + 16, 60, 348, t("做法", "How"), 16, MUT, True, "left", 24) + txt(x + 16, 90, 348, how, 18, TXT, False, "left", 130)
        o += txt(x + 16, 240, 348, t("風險", "Risk"), 16, MUT, True, "left", 24) + card(x + 16, 270, 348, 120, "red", .18) + txt(x + 24, 280, 332, risk, 18, TXT, False, "left", 100)
    o += txt(16, 410, 1128, t("先判斷變換是否保留類別，再選路線；局部共享只是其中一種架構先驗，三種可以結合。", "First decide whether the transformation preserves the class, then choose a route; local sharing is one architectural prior and the three can be combined."), 17, TXT, True, "center", 52)
    return o

# ------------------------------------------------ 6: geometry of products
def page6(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "blue", t("平移：位移向量相同的像素對可共享", "Translation: pixel pairs with the same displacement share"))
    s = ""
    for i in range(5):
        for j in range(5): s += f'<circle cx="{70+j*70}" cy="{110+i*62}" r="7" fill="#6b645a"/>'
    def pt(a, b): return 70 + b * 70, 110 + a * 62
    for (p, q, col) in (((0, 0), (1, 1), "#6a9fd6"), ((2, 2), (3, 3), "#6a9fd6"), ((0, 3), (2, 5 - 1), "#d97757")):
        (x1, y1), (x2, y2) = pt(*p), pt(*q); s += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="5"/><circle cx="{x1}" cy="{y1}" r="9" fill="{col}"/><circle cx="{x2}" cy="{y2}" r="9" fill="{col}"/>'
    o += svg(0, 0, W, H, s)
    o += txt(20, 386, 520, t("藍線位移都是 (1,1)：可共用係數；橘線位移 (2,2)：斜率相同卻不能共用。", "Both blue lines have displacement (1,1): they can share a coefficient; the orange one has displacement (2,2): same slope, but no sharing."), 16, TXT, False, "left", 76)
    o += card(20, 442, 520, 30, "yellow", .12) + txt(24, 444, 512, "(p_k + t) − (p_j + t) = p_k − p_j", 16, TXT, True, "center", 24)
    o += cardH(576, 0, 584, 480, "teal", t("旋轉：長度相同的像素對可共享", "Rotation: pixel pairs of equal length share"))
    s = f'<circle cx="760" cy="230" r="110" fill="none" stroke="#5fb8a8" stroke-width="3" stroke-dasharray="9 7"/><line x1="760" y1="230" x2="870" y2="230" stroke="#6a9fd6" stroke-width="6"/><line x1="760" y1="230" x2="760" y2="120" stroke="#d97757" stroke-width="6"/><circle cx="760" cy="230" r="8" fill="#f2ede4"/><circle cx="870" cy="230" r="10" fill="#6a9fd6"/><circle cx="760" cy="120" r="10" fill="#d97757"/>'
    o += svg(0, 0, W, H, s) + txt(790, 240, 90, "(1, 0)", 16, TXT, True, "center", 24) + txt(660, 90, 90, "(0, 1)", 16, TXT, True, "center", 24)
    o += txt(900, 120, 240, t("長度都是 1：旋轉保留長度，可讓相同長度的像素對共用二階係數。", "Both have length 1: rotation preserves length, so pairs of equal length can share a second-order coefficient."), 16, TXT, False, "left", 120)
    o += txt(596, 380, 548, t("在有限網格上，任意角度旋轉常需插值或裁切；只有相容的轉換、項與彙整，才可能精確不變。", "On a finite grid an arbitrary rotation often needs interpolation or cropping; exact invariance needs compatible transformations, terms and pooling."), 16, MUT, False, "left", 76)
    return o

# ------------------------------------------------ 7: square-term special case
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 480, "violet", t("平方項特例：v = b + Σaᵢxᵢ + cΣxᵢ²", "The square-term case: v = b + Σaᵢxᵢ + cΣxᵢ²"))
    o += card(16, 56, 668, 80, "violet", .22) + formula(16, 66, 668, r"v=b+\sum_i a_ix_i+c\sum_i x_i^{2}", 28, 60)
    o += txt(16, 148, 668, t("c 是所有平方項共同的係數，比原來只多一個參數，而不是每格各加一個。平方和就是 ‖x‖²。", "c is one coefficient shared by all square terms: one extra parameter, not one per cell. The sum of squares is ‖x‖²."), 16, TXT, False, "left", 52)
    for i, (pt_, sq, sc) in enumerate((("(1, 0)", "1² + 0² = 1", "b + 1 + c"), ("(0, 1)", "0² + 1² = 1", "b + c"))):
        y0 = 224 + i * 100; o += card(16, y0, 668, 90, "blue" if i == 0 else "orange", .18) + txt(28, y0 + 8, 130, pt_, 22, TXT, True, "left", 34)
        o += txt(160, y0 + 8, 240, t("平方和：" + sq, "sum of squares: " + sq), 17, TXT, False, "left", 34) + txt(160, y0 + 48, 520, t("a = (1, 0) 時的分數：" + sc, "score for a = (1, 0): " + sc), 18, TXT, True, "left", 34)
    o += cardH(716, 0, 444, 480, "orange", t("一個項對稱，不代表整體對稱", "One symmetric term does not make the model symmetric"))
    o += txt(732, 52, 412, t("兩點的平方和相同，所以平方和對正交旋轉不變。", "Both points have the same sum of squares, so the sum of squares is invariant under orthogonal rotation."), 17, TXT, False, "left", 76)
    o += card(732, 140, 412, 90, "red", .22) + txt(740, 148, 396, t("但線性項不同：分數 b + 1 + c 與 b + c 不相等", "But the linear term differs: the scores b + 1 + c and b + c are not equal"), 17, TXT, True, "center", 74)
    o += txt(732, 246, 412, t("歷史架構：三個隱藏層，前兩層是特徵圖；每個單元只接收前一層的局部視窗，同一張圖上的對應連線共用權重。", "The historical architecture: three hidden layers, the first two made of feature maps; each unit sees a local window of the previous layer and corresponding connections on a map share weights."), 16, MUT, False, "left", 120)
    o += card(732, 376, 412, 80, "yellow", .14) + txt(740, 384, 396, t("與本例的共享是同一個原理，但沒有足夠細節就不能推算完整參數數或準確率。", "Same principle as this example, but without enough detail its parameter count or accuracy cannot be derived."), 16, TXT, False, "center", 64)
    return o

# ------------------------------------------------ 8: exercises
def page8(lang):
    t = tr(lang); o = ""; n, pos, nos, sh = K.params(8); x = K.image(1); tx = K.image(3)
    qs = [("blue", t("題 1", "Q1"), t("8×8 輸入、3×3 視窗、步長 1、不補邊，偏置也共享：特徵圖多大？共享與不共享各有幾個參數？", "8×8 input, 3×3 window, stride 1, no padding, shared bias: how large is the feature map, and how many parameters with and without sharing?"), [rf"8-3+1={n},\ \ {n}\times{n}={pos}", rf"{pos}\times(9+1)={nos},\ \ 9+1={sh}"], t(f"{n}×{n} 特徵圖；不共享 {nos}，共享 {sh}。", f"{n}×{n} map; {nos} without sharing, {sh} with."), t("核對：影像變大只增加使用次數，不增加共享參數。", "Check: a larger image only adds uses, not shared parameters.")),
          ("red", t("題 2", "Q2"), t("共享後，高分位置跟著輸入移動，這叫等變還是不變？", "After sharing, the high-score position moves with the input: is this equivariance or invariance?"), [r"F(Tx)=T\,F(x)", r"F(Tx)\neq F(x)"], t("等變；再做適當彙整才可能不變。", "Equivariance; invariance needs a suitable pooling."), t("核對：本例最大值彙整後 z 都是 3。", "Check: after max pooling z is 3 in both cases."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        xx = i * 590; o += card(xx, 0, 570, 470, role) + head(xx, 0, 570, role, name, 36, 18)
        o += txt(xx + 16, 52, 538, q, 18, TXT, True, "left", 96)
        o += chip(xx + 16, 160, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(xx + 16, 204, 538, 140, "grey", .1) + formula(xx + 16, 212, 538, tex[0], 22, 56) + formula(xx + 16, 276, 538, tex[1], 22, 56)
        o += chip(xx + 16, 364, 538, role, ans, True, 42, 19)
        o += txt(xx + 16, 420, 538, chk, 17, MUT, False, "left", 48)
    return o

# ------------------------------------------------ 9: extension: exact window formula
def page9(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "blue", t("共享視窗的精確式（互相關，不翻轉核）", "The exact shared-window formula (cross-correlation, kernel not flipped)"))
    o += card(16, 56, 608, 100, "blue", .22) + formula(16, 66, 608, r"r_{ij}=b+\sum_{a=0}^{2}\sum_{c=0}^{2}K_{ac}\,x_{i+a,\,j+c}", 24, 80)
    o += txt(16, 170, 608, t("i, j 是視窗左上角；a, c 是核內的列、欄偏移（用 c 避免與偏置 b 混淆）。有逐點活化 φ 時 F(x)ᵢⱼ = φ(rᵢⱼ)；主例子直接用分數 r。", "i, j: top-left corner of the window; a, c: row and column offsets in the kernel. With an activation φ, F(x)ᵢⱼ = φ(rᵢⱼ); the main example uses r directly."), 16, TXT, False, "left", 124)
    o += card(16, 310, 608, 140, "grey", .12) + formula(16, 322, 608, r"\frac{\partial J}{\partial k}=\sum_p\frac{\partial J}{\partial r_p}\frac{\partial r_p}{\partial k}", 24, 70) + txt(24, 400, 592, t("局部線性計分下，第二個因子就是該位置的輸入 x_p。", "Under local linear scoring the second factor is just the input x_p at that position."), 16, MUT, False, "center", 40)
    o += cardH(656, 0, 504, 480, "orange", t("有限網格的邊界", "Boundaries of a finite grid"))
    for i, (a, b) in enumerate(((t("物件移出畫面", "The object leaves the frame"), t("等式可能只在內部區域成立", "equality may hold only inside")), (t("邊界補值不同、平移量與步長不相容", "Other padding, or shift vs stride"), t("對應位置不合法", "not all positions valid")), (t("影像旋轉", "Image rotation"), t("需要插值；相同長度的權重分組不能消除取樣誤差", "needs interpolation; equal-length grouping cannot remove sampling error")))):
        y0 = 54 + i * 122; o += card(672, y0, 472, 112, "grey", .12) + txt(684, y0 + 8, 448, a, 18, TXT, True, "left", 50) + txt(684, y0 + 62, 448, "→ " + b, 16, TXT, False, "left", 44)
    o += txt(672, 424, 472, t("平移等變性的公式需要對應位置都合法。", "Equivariance needs every position to be valid."), 16, MUT, False, "center", 40)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 10)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp12_" + fid, fn, W, H, out)
