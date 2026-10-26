"""One small example spec for each of the new layouts (bars_compare, tree_diagram, network_layers, quadrant_2x2, concept_card,
equation_parts); the other six are demonstrated by kp23_specs.py and kp12_specs.py.  Run: python3 layout_gallery.py outdir"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from figkit import layouts


def tr(zh, en): return {"zh": zh, "en": en}


GALLERY = {
 "bars_compare": {"id": "g-bars", "alt": tr("20 份訓練集的新點準確率：單棵樹 84.4±1.4、Bagging 87.7±1.1、隨機森林 87.8±1.1。", "Accuracy on new points over 20 training sets: single tree 84.4±1.4, bagging 87.7±1.1, random forest 87.8±1.1."),
   "unit": tr("新資料準確率（%，平均 ± 標準差）", "accuracy on new data (%, mean ± sd)"), "min": 80, "max": 92, "ticks": [80, 84, 88, 92],
   "bars": [{"label": tr("單棵樹", "single tree"), "value": 84.4, "err": 1.4, "role": "red"}, {"label": "Bagging", "value": 87.7, "err": 1.1, "role": "green", "note": tr("25 棵樹", "25 trees")}, {"label": tr("隨機森林", "random forest"), "value": 87.8, "err": 1.1, "role": "blue", "note": tr("每節點看 1 個特徵", "1 feature per node")}],
   "takeaway": tr("多棵樹投票：平均升高，散布變小。", "Voting trees: higher mean, smaller spread.")},
 "tree_diagram": {"id": "g-tree", "alt": tr("決策樹：x₂>0.5 為是→葉 A；否→再問 x₁≤0.4，是→葉 B、否→葉 C。", "Decision tree: x₂>0.5 yes → leaf A; no → ask x₁≤0.4, yes → leaf B, no → leaf C."),
   "root": {"text": "x₂ > 0.5 ?", "role": "blue", "children": [{"edge": tr("是", "yes"), "node": {"text": tr("葉 A", "leaf A"), "role": "blue"}}, {"edge": tr("否", "no"), "node": {"text": "x₁ ≤ 0.4 ?", "role": "orange", "children": [{"edge": tr("是", "yes"), "node": {"text": tr("葉 B", "leaf B"), "role": "orange"}}, {"edge": tr("否", "no"), "node": {"text": tr("葉 C", "leaf C"), "role": "green"}}]}}]},
   "takeaway": tr("每個問題只測一個特徵與門檻。", "Each question tests one feature against a threshold.")},
 "network_layers": {"id": "g-net", "alt": tr("2-3-1 的前饋網路：輸入 2 個、隱藏 3 個、輸出 1 個。", "A 2-3-1 feed-forward network: 2 inputs, 3 hidden units, 1 output."),
   "layers": [{"label": tr("輸入", "input"), "n": 2, "role": "blue", "formula": "$x$"}, {"label": tr("隱藏層", "hidden"), "n": 3, "role": "orange", "formula": "$h=f(Wx+b)$"}, {"label": tr("輸出", "output"), "n": 1, "role": "green", "formula": "$\\hat y=g(vh)$"}],
   "highlight": [[0, 0], [1, 1], [2, 0]], "takeaway": tr("每條線是一個權重；黃線是一條路徑。", "Every line is a weight; the yellow line is one path.")},
 "quadrant_2x2": {"id": "g-quad", "alt": tr("混淆矩陣：TP=40、FP=10、FN=5、TN=45。", "Confusion matrix: TP=40, FP=10, FN=5, TN=45."),
   "x_label": tr("實際", "actual"), "y_label": tr("預測", "predicted"), "x_vals": [tr("正", "positive"), tr("負", "negative")], "y_vals": [tr("正", "positive"), tr("負", "negative")],
   "cells": [{"role": "green", "title": "TP", "text": tr("預測正、實際正", "predicted + and actually +"), "value": "40"}, {"role": "red", "title": "FP", "text": tr("預測正、實際負（誤報）", "predicted + but actually − (false alarm)"), "value": "10"},
             {"role": "orange", "title": "FN", "text": tr("預測負、實際正（漏報）", "predicted − but actually + (miss)"), "value": "5"}, {"role": "blue", "title": "TN", "text": tr("預測負、實際負", "predicted − and actually −"), "value": "45"}],
   "takeaway": tr("準確率 = (40+45)/100 = 85%。", "Accuracy = (40+45)/100 = 85%.")},
 "concept_card": {"id": "g-card", "alt": tr("感受野：單元直接接收的輸入範圍；例：3×3 視窗；反例：整張影像。", "Receptive field: the input range a unit receives; example: a 3×3 window; non-example: the whole image."),
   "term": tr("感受野", "Receptive field"), "definition": tr("某個單元直接接收的輸入範圍。", "The range of input a unit directly receives."),
   "parts": [{"label": tr("例子", "example"), "text": tr("本例：影像中一塊 3×3 的視窗", "here: a 3×3 window of the image"), "role": "green"}, {"label": tr("反例", "non-example"), "text": tr("整張 6×6 影像（那是全連接）", "the whole 6×6 image (that is fully connected)"), "role": "red"}, {"label": tr("為什麼", "why"), "text": tr("限制範圍，才能讓各位置共用同一組權重", "limiting the range lets positions share one set of weights"), "role": "blue"}],
   "formula": "$\\text{positions}=(6-3+1)^2=16$", "takeaway": tr("先看範圍多大，再談共享。", "First the size of the range, then sharing.")},
 "equation_parts": {"id": "g-eq", "alt": tr("ΔI = I(t) − 0.4×I(tY) − 0.6×I(tN) = 1.522 − 0.325 − 0.875 = 0.322。", "ΔI = I(t) − 0.4×I(tY) − 0.6×I(tN) = 1.522 − 0.325 − 0.875 = 0.322."), "caption": tr("不純度下降量的三個部分", "the three parts of the impurity decrease"),
   "pieces": [{"tex": "$\\Delta I$", "role": "green", "name": tr("下降量", "decrease"), "meaning": tr("越大越好", "larger is better"), "value": "$0.322$"}, {"tex": "$=$"}, {"tex": "$I(t)$", "role": "blue", "name": tr("分裂前", "before"), "meaning": tr("父節點的不純度", "parent impurity"), "value": "$1.522$"}, {"tex": "$-$"},
              {"tex": "$\\frac{N_{tY}}{N_t}I(t_Y)$", "role": "orange", "name": tr("「是」堆", "yes pile"), "meaning": tr("占比 × 不純度", "share × impurity"), "value": "$0.325$"}, {"tex": "$-$"}, {"tex": "$\\frac{N_{tN}}{N_t}I(t_N)$", "role": "red", "name": tr("「否」堆", "no pile"), "meaning": tr("占比 × 不純度", "share × impurity"), "value": "$0.875$"}],
   "takeaway": tr("每一堆都乘上自己的點數占比。", "Each pile is weighted by its share of the points.")},
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "gallery"
    os.makedirs(out, exist_ok=True)
    for name, spec in GALLERY.items():
        layouts.build(name, spec).to_png(os.path.join(out, name + ".png")); print("png", name)
