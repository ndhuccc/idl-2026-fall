"""KP10 (network size, generalization, early stopping) -- third sample of view-from-text-v2. Plan: kp10_plan.json."""
import os, sys
from math import comb
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit.parts import T, NFig, header_chip, panel, chip, outline_box, icon, takeaway, tint, solid, INK


def tr(zh, en): return {"zh": zh, "en": en}


def generalization():
    rng = np.random.default_rng(3); xs = np.linspace(0.05, 0.95, 15); ys = np.sin(2 * np.pi * xs) + rng.normal(0, 0.3, 15)
    curve = [(float(x), float(np.sin(2 * np.pi * x))) for x in np.linspace(0, 1, 61)]
    pts = [(float(a), float(b)) for a, b in zip(xs, ys)]
    spec = {
        "id": "lec05-kp10-trend",
        "alt": tr("教學示意（不是網路訓練結果）：15 個帶雜訊的圓點散在一條平滑曲線上下；折線穿過每個點，背熟了雜訊，不是學到規律。泛化：對沒參與訓練的新資料仍能正確預測。欠擬合：連主要規律都表達不好。過擬合：配合訓練資料的偶然細節，損害新資料的預測。",
                  "Teaching illustration (not a network training result): 15 noisy dots scatter around a smooth curve; the zigzag line passes through every dot, so it memorised the noise instead of learning the pattern. Generalization: still predicting new data correctly. Underfitting: not even the main pattern is expressed. Overfitting: fitting the chance details of the training data, which hurts predictions on new data."),
        "plot": {"x": [0, 1], "y": [-1.7, 1.7], "x_ticks": [0, 0.5, 1], "y_ticks": [-1, 0, 1], "x_label": tr("輸入 x（教學示意，不是網路訓練結果）", "input x (teaching illustration, not a network result)"), "y_label": "y", "w": 520, "h": 300},
        "series": [{"kind": "line", "points": curve, "role": "green", "width": 4, "name": tr("平滑規律", "smooth pattern")},
                   {"kind": "line", "points": pts, "role": "red", "width": 2, "name": tr("折線：穿過每個點", "zigzag (every dot)")},
                   {"kind": "dots", "points": pts, "role": "orange", "d": 12, "name": tr("帶雜訊的樣本", "noisy samples")}],
        "side": [{"title": tr("泛化", "Generalization"), "role": "green", "name": tr("term:泛化", "term:Generalization"), "text": tr("對沒參與訓練的新資料\n仍能正確預測；\n不是把原來的答案背熟。", "still predicts new data\ncorrectly; not memorising\nthe original answers.")},
                 {"title": tr("欠擬合", "Underfitting"), "role": "blue", "name": tr("term:欠擬合", "term:Underfitting"), "text": tr("模型連主要規律都表達不好，\n訓練與新資料都可能差。", "the model does not even express\nthe main pattern; both can be poor.")},
                 {"title": tr("過擬合", "Overfitting"), "role": "red", "name": tr("term:過擬合", "term:Overfitting"), "text": tr("模型配合訓練資料的偶然細節，\n反而損害新資料的預測。", "the model fits chance details of the\ntraining data and hurts new data.")}],
        "takeaway": tr("要學主要規律，不要把雜訊也背熟；「較大」不是過擬合的充分條件。", "Learn the main pattern, not the noise; “larger” alone does not imply overfitting."),
    }
    return "plot_xy", spec


def parameters():
    f = NFig("lec05-kp10-params", 1160, 420, T(tr("2-3-1 全相連網路的自由參數：隱藏層每個單元 2 個權重加 1 個偏置，(2+1)×3=9；輸出層 3 個權重加 1 個偏置，(3+1)×1=4；總共 9+4=13 個，不是只有三個隱藏單元。",
                                                   "Free parameters of a fully connected 2-3-1 network: each hidden unit has 2 weights plus 1 bias, (2+1)×3=9; the output unit has 3 weights plus 1 bias, (3+1)×1=4; 9+4=13 in all, not just the three hidden units.")))
    header_chip(f, 0, 0, 640, 34, "blue", T(tr("2-3-1 全相連網路（每個計算單元有偏置）", "Fully connected 2-3-1 network (every unit has a bias)")), size=18)
    panel(f, 0, 38, 640, 300, "blue")
    xs = (110, 320, 530); ys = lambda n, k: 188 + (k - (n - 1) / 2) * 62
    for a in range(2):
        for b in range(3): f.line(xs[0], ys(2, a), xs[1], ys(3, b), "GAP", 2)
    for b in range(3): f.line(xs[1], ys(3, b), xs[2], ys(1, 0), "GAP", 2)
    for i, (n, role) in enumerate(((2, "blue"), (3, "orange"), (1, "green"))):
        for k in range(n): f.ellipse(xs[i], ys(n, k), 34, solid(role))
    f.text(xs[0], 90, T(tr("輸入 2 個", "2 inputs")), "TXT", 18, True); f.text(xs[1], 90, T(tr("隱藏 3 個", "3 hidden")), "TXT", 18, True); f.text(xs[2], 90, T(tr("輸出 1 個", "1 output")), "TXT", 18, True)
    chip(f, 150, 290, 260, 30, "orange", T(tr("每個隱藏單元：2 權重 ＋ 1 偏置", "per hidden unit: 2 weights + 1 bias")), size=16)
    f.text(xs[2], 236, T(tr("3 權重 ＋ 1 偏置", "3 weights + 1 bias")), "TXT", 16, True, w=150)
    header_chip(f, 680, 0, 480, 34, "green", T(tr("自由參數", "Free parameter")), size=20)
    panel(f, 680, 38, 480, 300, "green")
    f.text(698, 84, T(tr("訓練時可以獨立改變的數字；\n權重與偏置都算。", "numbers that can be changed independently\nin training; weights and biases both count.")), "TXT", 18, False, "left", w=450, name=T(tr("term:自由參數", "term:Free parameter")))
    f.text(920, 160, T(tr("隱藏層：$(2+1)\\times3=9$", "hidden layer: $(2+1)\\times3=9$")), "TXT", 22, w=440, math=True)
    f.text(920, 204, T(tr("輸出層：$(3+1)\\times1=4$", "output layer: $(3+1)\\times1=4$")), "TXT", 22, w=440, math=True)
    f.rect(740, 236, 360, 56, fill=tint("green", 0.30), radius=12, text="$9+4=13$", tcolor="TXT", size=28, bold=True, math=True)
    f.text(920, 316, T(tr("不是只有三個隱藏單元", "not just the three hidden units")), "TXT", 16, w=440)
    takeaway(f, 0, 356, 1160, 44, T(tr("先數清楚有多少可調的數字，再談模型太大還是太小。", "First count the adjustable numbers, then ask whether the model is too large or too small.")))
    return f


def early_stopping():
    ep = [1, 2, 3, 4, 5]; tr_l = [0.50, 0.30, 0.20, 0.15, 0.10]; va = [0.55, 0.35, 0.25, 0.28, 0.40]
    best = va.index(min(va)) + 1
    spec = {
        "id": "lec05-kp10-early",
        "alt": tr(f"五個檢查點的訓練與驗證損失（手編教學數字，不是實驗結果）：訓練 {', '.join(f'{v:.2f}' for v in tr_l)}；驗證 {', '.join(f'{v:.2f}' for v in va)}。驗證最低 {min(va):.2f} 在第 {best} 輪，保留第 {best} 輪；第 5 輪訓練最低但驗證升到 {va[4]:.2f}，增加 {va[4]-min(va):.2f}。",
                  f"Training and validation loss at five checkpoints (hand-made teaching numbers, not experiment results): training {', '.join(f'{v:.2f}' for v in tr_l)}; validation {', '.join(f'{v:.2f}' for v in va)}. The validation minimum {min(va):.2f} is at epoch {best}, so keep epoch {best}; epoch 5 has the lowest training loss but validation rises to {va[4]:.2f}, an increase of {va[4]-min(va):.2f}."),
        "plot": {"x": [0.6, 5.4], "y": [0, 0.7], "x_ticks": ep, "y_ticks": [0, 0.2, 0.4, 0.6], "x_label": tr("訓練輪數（五個檢查點；手編教學數字）", "epoch (five checkpoints; hand-made teaching numbers)"), "y_label": tr("平均損失", "mean loss"), "w": 520, "h": 300},
        "series": [{"kind": "vline", "x": best, "role": "green", "width": 2, "label": tr(f"保留第 {best} 輪", f"keep epoch {best}")},
                   {"kind": "line", "points": list(zip(ep, tr_l)), "role": "blue", "markers": True, "labels": [f"{v:.2f}" for v in tr_l], "label_dy": 22, "name": tr("訓練損失", "training loss")},
                   {"kind": "line", "points": list(zip(ep, va)), "role": "orange", "markers": True, "labels": [f"{v:.2f}" for v in va], "label_dy": -22, "name": tr("驗證損失", "validation loss")}],
        "side": [{"title": tr("驗證集", "Validation set"), "role": "green", "name": tr("term:驗證集", "term:Validation set"), "text": tr("不拿來更新權重，\n用來選模型或訓練檢查點的資料。", "not used to update weights;\nused to choose the model or checkpoint.")},
                 {"title": tr("過度訓練", "Overtraining"), "role": "red", "name": tr("term:過度訓練", "term:Overtraining"), "text": tr("架構固定，後續訓練追逐訓練資料的細節，\n驗證表現反而惡化。", "with a fixed architecture, later training\nchases training-data details and validation worsens.")},
                 {"title": tr("早停", "Early stopping"), "role": "blue", "name": tr("term:早停", "term:Early stopping"), "text": tr("根據驗證表現停止訓練，\n並還原最佳檢查點。", "stop training by validation performance\nand restore the best checkpoint.")}],
        "takeaway": tr(f"第 5 輪訓練損失最低，驗證卻從 {min(va):.2f} 升到 {va[4]:.2f}（增加 {va[4]-min(va):.2f}）：保留第 {best} 輪。", f"Epoch 5 has the lowest training loss but validation rises from {min(va):.2f} to {va[4]:.2f} (+{va[4]-min(va):.2f}): keep epoch {best}."),
    }
    return "plot_xy", spec


def data_roles():
    spec = {
        "id": "lec05-kp10-roles",
        "alt": tr("三種資料：訓練集用來更新權重；驗證集用來選大小、懲罰強度與停止點，不直接更新權重；測試集在模型選完之後才做最後評估，反覆看它來挑模型就不再獨立。小樣本的數字很粗：十筆中錯一筆是 1/10=0.1，錯兩筆是 2/10=0.2。",
                  "Three kinds of data: the training set updates the weights; the validation set picks size, penalty strength and stopping point without updating weights; the test set is used only for the final evaluation once the model is chosen, and looking at it repeatedly to choose stops it being independent. Small samples are coarse: one error in ten is 1/10=0.1, two are 2/10=0.2."),
        "columns": [
            {"role": "blue", "header": tr("訓練集", "Training set"), "items": [{"label": tr("用途", "role"), "lines": [tr("用它更新權重，", "weights are updated on it;"), tr("學到資料中的規律", "it learns the pattern")], "name": tr("term:訓練集", "term:Training set")}, {"label": tr("不能", "not for"), "lines": [tr("用它的分數證明新資料也準", "proving the model is good on new data")]}]},
            {"role": "green", "header": tr("驗證集", "Validation set"), "items": [{"label": tr("用途", "role"), "lines": [tr("選大小、懲罰強度", "choose size, penalty"), tr("與停止點", "and stopping point")]}, {"label": tr("不能", "not for"), "lines": [tr("不直接更新權重", "it does not update weights")]}]},
            {"role": "orange", "header": tr("測試集", "Test set"), "items": [{"label": tr("用途", "role"), "lines": [tr("模型選完之後，", "after the model is chosen,"), tr("才做最後評估", "the final evaluation")], "name": tr("term:測試集", "term:Test set")}, {"label": tr("注意", "careful"), "lines": [tr("反覆看它挑模型就不再獨立；", "reusing it to pick models ends independence;"), tr("避免重複個體或時間資訊洩漏", "avoid duplicate people, time leakage")]}]}],
        "takeaway": tr("小樣本的數字很粗：十筆中錯一筆是 1/10=0.1，錯兩筆是 2/10=0.2。", "Small samples are coarse: one error in ten is 1/10=0.1, two are 2/10=0.2."),
    }
    return "columns_text", spec


def capacity():
    ks = [0, 1, 2, 3, 4]
    M = lambda K, d: sum(comb(K, m) for m in range(d + 1))
    def xor(f, x, y, w, h, lang):
        cx, cy, s = x + w / 2, y + h / 2 - 24, 52
        outline_box(f, cx - 90, cy - 52, 180, 104, "MUT", 1.5)
        for k, (a, b, kind) in enumerate(((-1, 1, "circle"), (1, -1, "circle"), (-1, -1, "square"), (1, 1, "square"))):
            icon(f, kind, cx + a * 46, cy + b * 30)
        f.line(cx - 90, cy - 29, cx + 34, cy + 52, "#7dbb8c", 3); f.line(cx - 34, cy - 52, cx + 90, cy + 29, "#7dbb8c", 3)
        f.text(cx, y + h - 6, T(tr("兩條平行線只切成三塊", "2 parallel lines: 3 regions"), lang), "TXT", 16, True, w=w - 16)
    spec = {
        "id": "lec05-kp10-regions",
        "alt": tr("K 個一般位置的超平面在 d 維空間最多切成 M(K,d)=Σ C(K,m) 塊：d=1 為 " + "、".join(str(M(k, 1)) for k in range(1, 5)) + "；d=2 為 " + "、".join(str(M(k, 2)) for k in range(1, 5)) + "；d=3 為 " + "、".join(str(M(k, 3)) for k in range(1, 5)) + "（K=1 到 4）。例：d=2、K=2：1+2+1=4。上界只是上限；XOR 的兩條平行線實際只切成三塊。",
                  "K hyperplanes in general position cut d-dimensional space into at most M(K,d)=Σ C(K,m) regions: d=1 gives " + ", ".join(str(M(k, 1)) for k in range(1, 5)) + "; d=2 gives " + ", ".join(str(M(k, 2)) for k in range(1, 5)) + "; d=3 gives " + ", ".join(str(M(k, 3)) for k in range(1, 5)) + " (K=1 to 4). Example d=2, K=2: 1+2+1=4. The bound is only a ceiling; the two parallel lines of XOR cut only three regions."),
        "plot": {"x": [0.6, 4.4], "y": [0, 16.5], "x_ticks": [1, 2, 3, 4], "y_ticks": [0, 5, 10, 15], "x_label": tr("超平面數 K", "number of hyperplanes K"), "y_label": tr("最多區域數", "most regions"), "w": 400, "h": 290, "legend_pos": "left"},
        "series": [{"kind": "line", "points": [(k, M(k, d)) for k in range(1, 5)], "role": r, "markers": True, "labels": [str(M(k, d)) if (d == 1 or (d == 3 and k == 2) or k >= 3) else None for k in range(1, 5)], "label_dy": -20, "name": tr(f"d = {d}", f"d = {d}")} for d, r in ((1, "blue"), (2, "green"), (3, "orange"))],
        "side": [{"title": tr("區域數上界", "Cell-count bound"), "role": "blue", "name": tr("term:區域數上界", "term:Cell-count bound"), "h": 1.55,
                  "text": tr("$K$ 個超平面切 $d$ 維空間，區域數最多\n$M_{\\max}(K,d)=\\sum_{m=0}^{d}\\binom{K}{m}$\n例：$d=2,K=2$：$1+2+1=4$", "$K$ hyperplanes cut $d$-space into at most\n$M_{\\max}(K,d)=\\sum_{m=0}^{d}\\binom{K}{m}$\nexample: $d=2,K=2$: $1+2+1=4$")},
                 {"title": tr("超平面", "Hyperplane"), "role": "green", "name": tr("term:超平面", "term:Hyperplane"), "h": 1.1, "text": tr("二維是直線、三維是平面；\n一個隱藏單元的分界就是一個超平面。", "a line in 2D, a plane in 3D; the boundary\nof one hidden unit is one hyperplane.")},
                 {"title": tr("XOR 的實際構造", "XOR in practice"), "role": "orange", "h": 1.5, "draw": xor}],
        "takeaway": tr("上界只是上限，不是品質保證：兩條平行線實際只切成三塊，不是一般位置的最大值四。", "A ceiling, not a guarantee: two parallel lines cut only three regions, not the maximum of four."),
    }
    return "plot_xy", spec


def strategies():
    spec = {
        "id": "lec05-kp10-strategies",
        "alt": tr("三種選大小的策略：解析法（用代數或統計估計能力，通常忽略最佳化過程）、剪枝（先訓練較大的網路再刪連線，下一個 KP）、構造式方法（從小架構逐步增加單元，KP05），都要用驗證資料檢查；來源正弦示範：15 個訓練點、10 個測試點，模型參數數 3k+1，來源描述 k 約 3 到 5 較合適、k=15 可能過擬合，不是通用最佳值，本教材沒有重跑。",
                  "Three strategies to choose the size: analytic methods (estimate capacity with algebra or statistics, usually ignoring the optimisation), pruning (train a larger network, then delete connections; next KP), constructive methods (grow from a small architecture, KP05); all need validation data. The source sine demo: 15 training points, 10 test points, 3k+1 parameters; the source describes k of about 3 to 5 as suitable and k=15 as possibly overfitting; not a universal optimum, and this lecture did not rerun it."),
        "columns": [
            {"role": "blue", "header": tr("解析法", "Analytic"), "items": [{"label": tr("做法", "how"), "lines": [tr("用代數或統計", "estimate capacity with"), tr("估計能力", "algebra or statistics")]}, {"label": tr("風險", "risk"), "lines": [tr("通常忽略具體的", "usually ignores the"), tr("最佳化過程", "optimisation process")]}]},
            {"role": "green", "header": tr("剪枝", "Pruning"), "items": [{"label": tr("做法", "how"), "lines": [tr("先訓練較大的網路，", "train a larger network,"), tr("再刪較不重要的連線", "then delete weak connections")]}, {"label": tr("要檢查", "check"), "lines": [tr("用驗證資料（下一個 KP）", "with validation data (next KP)")]}]},
            {"role": "orange", "header": tr("構造式方法", "Constructive"), "items": [{"label": tr("做法", "how"), "lines": [tr("從小架構逐步", "grow a small"), tr("增加單元（KP05）", "architecture (KP05)")]}, {"label": tr("要檢查", "check"), "lines": [tr("用驗證資料", "with validation data")]}]},
            {"role": "red", "header": tr("來源的正弦示範", "The source sine demo"), "items": [{"label": tr("設定", "setup"), "lines": ["$y=\\sin(2\\pi x)+\\epsilon$", tr("15 個訓練點、10 個測試點", "15 training, 10 test points"), tr("參數數 $3k+1$", "parameters $3k+1$")]}, {"label": tr("來源描述", "source"), "lines": [tr("$k$ 約 3–5 較合適，", "$k$ about 3-5 suits,"), tr("$k=15$ 可能過擬合，", "$k=15$ may overfit,"), tr("非通用最佳值", "not a universal optimum")]}]}],
        "takeaway": tr("三條路都要用驗證資料檢查；來源示範的 k 值不是任何資料的通用最佳值（本教材沒重跑）。", "All three need validation data; the source's k is not a universal optimum (this lecture did not rerun it)."),
    }
    return "columns_text", spec


def solution():
    spec = {
        "id": "lec05-kp10-solution",
        "alt": tr("解答：題 1 驗證 0.45、0.25、0.30，最小 0.25，保留第 2 輪，第 3 輪比它高 0.05；題 2 二維三條一般位置直線 C(3,0)+C(3,1)+C(3,2)=1+3+3=7，三線平行只有四條帶，沒達到七塊上界。",
                  "Solution: Q1 validation 0.45, 0.25, 0.30, minimum 0.25, keep epoch 2, epoch 3 is 0.05 higher; Q2 three lines in general position in 2D: C(3,0)+C(3,1)+C(3,2)=1+3+3=7; three parallel lines give only four bands, short of the bound of seven."),
        "steps": [{"role": "blue", "title": tr("① 題 1：選檢查點", "① Q1: pick a checkpoint"), "lines": [tr("驗證 0.45、0.25、0.30", "validation 0.45, 0.25, 0.30"), tr("最小是 0.25（第 2 輪）", "minimum 0.25 (epoch 2)"), "$0.30-0.25=0.05$"], "result": tr("保留第 2 輪", "keep epoch 2")},
                  {"role": "green", "title": tr("② 題 2：區域數", "② Q2: regions"), "lines": ["$\\binom{3}{0}+\\binom{3}{1}+\\binom{3}{2}$", "$1+3+3$"], "result": "$M_{\\max}=7$"},
                  {"role": "orange", "title": tr("③ 三線平行", "③ three parallel lines"), "lines": [tr("只有四條帶", "only four bands"), tr("沒達到七塊上界", "short of the bound of seven")], "result": "$4<7$"}],
        "takeaway": tr("選檢查點時，幾何區域數沒有參與比較；上界只是上限。", "Geometric region counts take no part in choosing a checkpoint; the bound is only a ceiling."),
    }
    return "steps_flow", spec
