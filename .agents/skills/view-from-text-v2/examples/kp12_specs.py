"""KP12 (weight sharing and translation invariance) -- second acceptance sample of view-from-text-v2.
Figure pages as layout specs; every number is computed here. Page plan: kp12_plan.json."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import NFig, T, header_chip, panel, chip, outline_box, ring, takeaway, terms_strip, tint, solid, INK


def tr(zh, en): return {"zh": zh, "en": en}


def data():
    x = np.zeros((6, 6), int); x[1:4, 1] = 1
    tx = np.zeros((6, 6), int); tx[1:4, 3] = 1
    K = np.zeros((3, 3), int); K[:, 1] = 1
    corr = lambda im: np.array([[int((im[i:i + 3, j:j + 3] * K).sum()) for j in range(4)] for i in range(4)])
    return x, tx, K, corr(x), corr(tx)


def idea():
    x, tx, K, Fx, Ftx = data()
    r = int((x[1:4, 0:3] * K).sum())
    spec = {
        "id": "lec05-kp12-idea", "height": 458, "terms_y": 340,
        "alt": tr(f"6×6 影像 x 有一條三格直筆畫（第 1 欄第 1 到 3 列），3×3 共享核 K 中間欄為 1，窗口在第 1 列第 0 欄的分數 r={r}；x 向右平移 2 欄得到 Tx，筆畫在第 3 欄；要的是類別不變 c(Tx)=c(x)，不是每個數字都不變。",
                  f"A 6×6 image x holds a three-cell vertical stroke (column 1, rows 1 to 3); the shared 3×3 kernel K has 1s in its middle column; the window at row 1, column 0 scores r={r}; shifting x right by 2 columns gives Tx with the stroke in column 3; we want the class to stay the same, c(Tx)=c(x), not every number."),
        "rows": [
            {"items": [{"kind": "grid", "rows": 6, "cols": 6, "cell": 20, "cells": x.tolist(), "role": "orange", "window": (1, 0, 3, 3), "caption": tr("影像 x（黃框＝一個視窗）", "image x (yellow = window)")},
                       {"kind": "op", "text": "⊛", "w": 60},
                       {"kind": "grid", "rows": 3, "cols": 3, "cell": 24, "cells": K.tolist(), "role": "green", "show": True, "caption": tr("共享核 K", "kernel K")},
                       {"kind": "op", "text": "=", "w": 50},
                       {"kind": "chip", "text": f"$r_{{1,0}}={r}$", "role": "yellow", "w": 140, "caption": tr("這個視窗的分數", "window score")}]},
            {"items": [{"kind": "grid", "rows": 6, "cols": 6, "cell": 20, "cells": x.tolist(), "role": "orange", "caption": tr("x：筆畫在第 1 欄", "x: stroke in col 1")},
                       {"kind": "op", "text": "→", "sub": tr("平移 2 欄", "shift 2 cols"), "w": 130},
                       {"kind": "grid", "rows": 6, "cols": 6, "cell": 20, "cells": tx.tolist(), "role": "orange", "caption": tr("Tx：筆畫在第 3 欄", "Tx: stroke in col 3")},
                       {"kind": "op", "text": "⇒", "w": 60},
                       {"kind": "chip", "text": "$c(Tx)=c(x)$", "role": "green", "w": 190, "caption": tr("要的是類別不變", "class unchanged")}]}],
        "terms": [(tr("轉換不變性", "Transformation invariance"), tr("指定轉換 T 後，分類結果不變\nc(Tx)=c(x)", "class unchanged after T:\nc(Tx) = c(x)")),
                  (tr("感受野", "Receptive field"), tr("單元直接接收的輸入範圍\n本例是 3×3 視窗", "input range of a unit:\nhere a 3×3 window")),
                  (tr("權重共享", "Weight sharing"), tr("多條連線共用同一個數值\n更新時也只更新這一個", "many connections share\none value, updated once")),
                  (tr("卷積核", "Kernel"), tr("滑動視窗重用的權重矩陣\n本節記為 K", "weight matrix reused by\nthe sliding window: K"))],
        "takeaway": tr("同一個 3×3 核重複用在每個位置；要的是「類別不變」，不是每個數字都不變。", "One 3×3 kernel is reused at every position; the goal is an unchanged class, not unchanged numbers."),
    }
    return "grid_flow", spec


def maps():
    x, tx, K, Fx, Ftx = data()
    row = lambda im, F, nm, cap: {"items": [
        {"kind": "grid", "rows": 6, "cols": 6, "cell": 20, "cells": im.tolist(), "role": "orange", "caption": tr(nm, nm.replace("影像", "image"))},
        {"kind": "op", "text": "→", "sub": tr("套用核 K", "apply K"), "w": 110},
        {"kind": "grid", "rows": 4, "cols": 4, "cell": 30, "cells": F.tolist(), "role": "blue", "show": True, "scale": 3, "caption": tr(cap, cap)},
        {"kind": "op", "text": "→", "sub": tr("取最大值", "take the max"), "w": 120},
        {"kind": "chip", "text": f"$z={int(F.max())}$", "role": "green", "w": 120, "caption": tr("讀出", "readout")}]}
    spec = {
        "id": "lec05-kp12-maps", "height": 458, "terms_y": 340,
        "alt": tr(f"同一個核 K 套用到 x 與平移後的 Tx：特徵圖 F(x) 的亮處在第 0 欄（{Fx[:,0].tolist()}），F(Tx) 的亮處在第 2 欄（{Ftx[:,2].tolist()}）；兩張特徵圖的最大值都是 z={int(Fx.max())}。反應跟著筆畫移動（等變），取最大值後的讀出相同（不變）。",
                  f"The same kernel K applied to x and to the shifted Tx: the bright column of F(x) is column 0 ({Fx[:,0].tolist()}) and that of F(Tx) is column 2 ({Ftx[:,2].tolist()}); both maps have maximum z={int(Fx.max())}. The response moves with the stroke (equivariance); the max readout is the same (invariance)."),
        "rows": [row(x, Fx, "影像 x", "F(x)：亮處在第 0 欄"), row(tx, Ftx, "影像 Tx", "F(Tx)：亮處在第 2 欄")],
        "terms": [(tr("特徵圖", "Feature map"), tr("每個視窗的分數\n依位置排成的陣列 F(x)", "scores of all windows\nby position, F(x)")),
                  (tr("平移等變性", "Translation equivariance"), tr("輸入平移，輸出跟著平移\nF(Tx) = T F(x)", "shift the input, the output\nshifts: F(Tx) = T F(x)")),
                  (tr("最大值彙整", "Global max pooling"), tr("整張特徵圖只取最大值\nz = max F(x)", "keep only the maximum\nof the map: z = max F(x)"))],
        "takeaway": tr("特徵圖跟著筆畫移動（等變）；只取最大值，讀出才不隨位置改變（不變）。", "The map moves with the stroke (equivariant); only after taking the maximum is the readout position-free (invariant)."),
    }
    return "grid_flow", spec


def counting():
    spec = {
        "id": "lec05-kp12-count",
        "alt": tr("參數計數：6×6 影像、3×3 視窗、步長 1、不補邊，共 4×4=16 個位置。不共享：16×(9+1)=160 個參數；共享：9+1=10 個參數，運算仍是 16 次。共享梯度 ∂J/∂k=Σ δp·xp；例：0.2×3+(−0.1)×1=0.5；k=1−0.1×0.5=0.95。",
                  "Counting parameters: 6×6 image, 3×3 window, stride 1, no padding, 4×4=16 positions. Not shared: 16×(9+1)=160 parameters; shared: 9+1=10 parameters, still 16 computations. Shared gradient ∂J/∂k=Σ δp·xp; example: 0.2×3+(−0.1)×1=0.5; k=1−0.1×0.5=0.95."),
        "panels": [
            {"role": "red", "header": tr("不共享：每個位置各有一組", "Not shared: every position has its own set"),
             "sketch": [("grid_tags", {"cx": 280, "y": 50, "n": 4, "tag": "10", "role": "red"})],
             "caption": tr("16 個位置，各有 9 個權重＋1 個偏置", "16 positions, each with 9 weights + 1 bias"),
             "stats": [{"label": tr("參數：", "parameters: "), "value": "160", "sub": "16 × (9 + 1)", "role": "red"}, {"label": tr("計算：", "computations: "), "value": tr("16 次", "16"), "sub": tr("4×4 個視窗", "4×4 windows"), "role": "grey"}],
             "gap": tr("每個位置各自調整", "each position tuned on its own")},
            {"role": "green", "header": tr("共享：所有位置共用一組", "Shared: all positions use one set"),
             "sketch": [("grid_tags", {"cx": 880, "y": 50, "n": 4, "tag": "K", "role": "green"})],
             "caption": tr("16 個位置，共用同一組 9＋1", "16 positions, one shared 9 + 1"),
             "stats": [{"label": tr("參數：", "parameters: "), "value": "10", "sub": "9 + 1", "role": "green"}, {"label": tr("計算：", "computations: "), "value": tr("16 次", "16"), "sub": tr("照算 16 次，只是共用數值", "still 16, just shared values"), "role": "grey"}],
             "gap": tr("少了 150 個可調數字，計算次數沒減", "150 fewer adjustable numbers, same amount of computing")}],
        "boxes": [
            {"kind": "text", "x": 0, "w": 376, "title": tr("共享梯度", "Shared gradient"), "title_w": 170, "math": True,
             "lines": ["$\\partial J/\\partial k=\\sum_p\\delta_p x_p$", tr("各位置的 δp·xp 加起來", "add δp·xp over all positions")]},
            {"kind": "text", "x": 392, "w": 376, "title": tr("數字例子", "Numeric example"), "title_w": 170, "math": True,
             "lines": ["$0.2\\times3+(-0.1)\\times1=0.5$", tr("兩個位置的貢獻相加", "two positions add up")]},
            {"kind": "text", "x": 784, "w": 376, "title": tr("更新", "Update"), "title_w": 120, "math": True,
             "lines": ["$k=1-0.1\\times0.5=0.95$", tr("所有位置接著都用 0.95", "every position then uses 0.95")]}],
        "takeaway": tr("共享省的是可獨立調整的數字（160→10），不是計算次數（仍是 16 次）。", "Sharing saves adjustable numbers (160 → 10), not computing (still 16)."),
        "footnote": tr("6×6 影像、3×3 視窗、步長 1、不補邊；學習率 η = 0.1，原 k = 1。", "6×6 image, 3×3 window, stride 1, no padding; learning rate η = 0.1, old k = 1."),
    }
    return "before_after", spec


def routes():
    spec = {
        "id": "lec05-kp12-routes",
        "alt": tr("讓分類不隨變換改變的三條路線：選不變特徵（風險：丟掉辨識所需資訊）、架構約束如權重共享加彙整（風險：只對相容的變換成立）、資料擴增（風險：有限例子不保證所有變換下不變）。", "Three routes to make a classifier insensitive to a transformation: invariant features (risk: lose information needed to recognise), architecture constraints such as weight sharing plus pooling (risk: valid only for compatible transformations), data augmentation (risk: finite examples do not guarantee invariance under all transformations)."),
        "columns": [
            {"role": "blue", "header": tr("選不變特徵", "Invariant features"),
             "items": [{"label": tr("做法", "how"), "lines": [tr("先把資料轉成對指定變換", "turn the data into a form"), tr("不敏感的表示", "insensitive to the transformation")]},
                       {"label": tr("風險", "risk"), "lines": [tr("可能丟掉辨識所需的資訊", "may lose information needed to recognise")]}]},
            {"role": "green", "header": tr("架構約束", "Architecture constraint"),
             "items": [{"label": tr("做法", "how"), "lines": [tr("共享權重加適當彙整，", "share weights and pool,"), tr("明確限制哪些計算必須相同", "forcing certain computations to be equal")]},
                       {"label": tr("風險", "risk"), "lines": [tr("只對相容的變換成立", "valid only for compatible transformations")]}]},
            {"role": "orange", "header": tr("資料擴增", "Data augmentation"),
             "items": [{"label": tr("做法", "how"), "lines": [tr("用合理的變換產生", "generate more training"), tr("更多訓練例子", "examples with sensible transformations")]},
                       {"label": tr("風險", "risk"), "lines": [tr("有限例子通常不保證", "finite examples rarely guarantee"), tr("所有變換下都不變", "invariance under every transformation")]}]}],
        "takeaway": tr("三條路線可以並用；先判斷這個變換是否保留類別。", "The routes can be combined; first decide whether the transformation keeps the class."),
    }
    return "columns_text", spec


def geometry():
    """Freeform figure: three panels (translation, rotation, a square term). Hand-drawn with parts."""
    f = NFig("lec05-kp12-geometry", 1160, 442,
             T(tr("像素兩兩相乘的幾何共享：平移時兩端點同加位移 t，差向量不變，位移相同 (1,1) 的線段可共享、(2,2) 的不可；旋轉保留線段長度，相同長度的像素對可共享；平方和 ‖x‖² 對旋轉不變，但線性項 a=(1,0) 讓 (1,0) 與 (0,1) 的分數 b+1+c 與 b+c 不同，整個模型並不對稱。",
                       "Geometric sharing of pixel products: under a shift both ends move by t so the difference vector is unchanged; segments with displacement (1,1) can share, (2,2) cannot; rotation keeps length, so pairs of equal length can share; the squared norm ‖x‖² is rotation-invariant, but a linear term a=(1,0) gives (1,0) and (0,1) scores b+1+c and b+c, so the whole model is not symmetric."), None))
    cw, gap = 372, 22
    heads = [("blue", tr("平移：位移相同才能共享", "Shift: same displacement shares")), ("green", tr("旋轉：長度相同才能共享", "Rotation: same length shares")), ("red", tr("平方項：一個項對稱，整體不一定", "Square term: not the whole model"))]
    for k, (role, h) in enumerate(heads):
        x = k * (cw + gap)
        header_chip(f, x, 0, cw, 38, role, T(h), size=18)
        panel(f, x, 44, cw, 300, role)
    # A: lattice with segments
    ax0, ay0, s = 70, 70, 44
    for i in range(5):
        for j in range(5):
            f.ellipse(ax0 + i * s, ay0 + j * s, 10, "#b7b0a5")
    P = lambda i, j: (ax0 + i * s, ay0 + j * s)
    for (a, b) in (((0, 0), (1, 1)), ((2, 1), (3, 2))): f.line(*P(*a), *P(*b), "#6a9fd6", 5)
    f.line(*P(0, 2), *P(2, 4), "#d97757", 5)
    f.text(250, 100, T(tr("位移 (1,1)", "shift (1,1)")), "#6a9fd6", 16, True, "left", w=130)
    f.text(250, 128, T(tr("兩條可共享", "these 2 share")), "TXT", 16, False, "left", w=130)
    f.text(250, 190, T(tr("位移 (2,2)", "shift (2,2)")), "#d97757", 16, True, "left", w=130)
    f.text(250, 218, T(tr("斜率相同，", "same slope,")), "TXT", 16, False, "left", w=130)
    f.text(250, 238, T(tr("但不能共用", "not shareable")), "TXT", 16, False, "left", w=130)
    f.text(186, 306, T(tr("兩端同加 t，差向量不變", "add t to both ends: same difference")), "TXT", 16, w=350)
    # B: ring with two points
    bx, by, r = cw + gap + 186, 170, 66
    ring(f, bx, by, r, "#7dbb8c", 3)
    f.line(bx, by, bx + r, by, "MUT", 2); f.line(bx, by, bx, by - r, "MUT", 2)
    f.ellipse(bx, by, 8, "TXT"); f.ellipse(bx + r, by, 17, "#6a9fd6"); f.ellipse(bx, by - r, 17, "#6a9fd6")
    f.text(bx + r + 12, by + 22, "(1,0)", "TXT", 16, False, "left", w=70); f.text(bx + 14, by - r - 14, "(0,1)", "TXT", 16, False, "left", w=70)
    f.text(bx + r / 2, by + 20, T(tr("長度 1", "length 1")), "#7dbb8c", 16, True, w=90)
    f.text(bx, 306, T(tr("旋轉保留長度：同長的像素對可共享", "rotation keeps length: pairs share")), "TXT", 16, w=350)
    # C: scores
    cx0 = 2 * (cw + gap)
    f.text(cx0 + 186, 80, "$\\|x\\|^2$：$(1,0)$ " + T(tr("與", "and")) + " $(0,1)$ → $1$", "TXT", 18, w=350, math=True)
    chip(f, cx0 + 16, 116, 340, 30, "green", T(tr("平方和對旋轉不變 ✓", "squared norm is rotation-invariant ✓")), size=16, bold=True)
    f.text(cx0 + 186, 176, T(tr("加上線性項 $a=(1,0)$：", "add a linear term $a=(1,0)$:")), "TXT", 16, w=350, math=True)
    f.text(cx0 + 186, 214, "$v(1,0)=b+1+c$", "TXT", 18, w=300, math=True)
    f.text(cx0 + 186, 248, "$v(0,1)=b+c$", "TXT", 18, w=300, math=True)
    chip(f, cx0 + 16, 276, 340, 30, "red", T(tr("兩點分數不同 ✗：整個模型不對稱", "scores differ ✗: model not symmetric")), size=16, bold=True)
    terms_strip(f, 352, [(T(tr("高階網路", "Higher-order network")), T(tr("活化前的分數含輸入乘積：\n$v=b+\\sum a_ix_i+\\sum c_{jk}x_jx_k$", "the pre-activation score contains products of inputs:\n$v=b+\\sum a_ix_i+\\sum c_{jk}x_jx_k$")))], h=90)
    return f


def solution():
    spec = {
        "id": "lec05-kp12-solution",
        "alt": tr("練習解答：8×8 影像、3×3 視窗、步長 1、不補邊：每個方向 8−3+1=6，共 6×6=36 個位置，特徵圖 6×6；不共享 36×(9+1)=360 個參數；共享 9+1=10 個；影像變大只增加使用次數，不增加共享參數。高分位置跟著輸入移動是等變。",
                  "Exercise solution: 8×8 image, 3×3 window, stride 1, no padding: 8−3+1=6 per direction, 6×6=36 positions, a 6×6 map; not shared 36×(9+1)=360 parameters; shared 9+1=10; a bigger image only adds uses, not shared parameters. High scores that follow the input are equivariance."),
        "steps": [
            {"role": "blue", "title": tr("① 位置數", "① positions"), "lines": ["$8-3+1=6$", tr("每個方向 6 個", "6 per direction"), "$6\\times6=36$"], "result": tr("特徵圖 6×6", "map 6×6")},
            {"role": "red", "title": tr("② 不共享", "② not shared"), "lines": ["$36\\times(9+1)$", tr("每個位置各 10 個", "10 for each position")], "result": "$=360$"},
            {"role": "green", "title": tr("③ 共享", "③ shared"), "lines": ["$9+1$", tr("影像變大只增加使用次數", "a bigger image only adds uses")], "result": "$=10$"}],
        "takeaway": tr("題 2：高分位置跟著輸入移動是「等變」；再做適當彙整且條件成立，才可能「不變」。", "Q2: a high score that follows the input is equivariance; pooling can make it invariance."),
    }
    return "steps_flow", spec
