import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP16"; N = 6
NOISE = 0.2; FLOOR = NOISE ** 2; assert abs(FLOOR - 0.04) < 1e-12
EXP = {3: (0.0260, 0.0533, 0.0571), 14: (0.0, 1072.06, 164.76), 24: (0.0, 0.5197, 0.8247)}   # (train, validation, test) MSE quoted from the note's experiment
assert min(EXP, key=lambda d: EXP[d][1]) == 3 and min(EXP, key=lambda d: EXP[d][0]) in (14, 24) and EXP[3][1] < EXP[3][2] and EXP[14][2] / EXP[3][2] > 2800

def make(lang):
    d = D(lang, KP, N); T = d.T
    g = d.page("KP16 其他收斂理論:條件式的光譜", "KP16 Other convergence theories")
    g.beat = 1
    chain(g, 50, 120, 1180, 190, [(BLUE, BLUEL, T("啟用函式", "activations"), T("二次啟用、ReLU、Leaky ReLU(線性可分資料)", "quadratic, ReLU, Leaky ReLU (linearly separable data)")), (ORG, ORGL, T("網路架構", "architectures"), T("殘差連接、非獨立同分布的資料流", "residual connections, non-i.i.d. data streams")), (PUR, PURL, T("幾何與界", "geometry and bounds"), T("臨界點的刻畫、交叉熵的隱式偏置", "characterizing critical points, the implicit bias of cross-entropy"))], ts=22, bs=20, gap=50, beat0=1)
    g.beat = 4
    g.rect(120, 340, 1040, 110, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(130, 340, 1020, 110, T("假設(寬度、初始化、資料、平滑性、步長)$\\ \\Longrightarrow\\ L_{\\mathrm{train}}(w_T)\\le\\varepsilon$", "assumptions (width, init, data, smoothness, step) $\\ \\Longrightarrow\\ L_{\\mathrm{train}}(w_T)\\le\\varepsilon$"), size=26, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 5
    gold(g, 150, 480, 980, 130, T("而不是「$\\Longrightarrow L_{\\mathrm{test}}$ 很小」。讀這類結果先問兩件事:它的假設我的問題滿足嗎?它保證的是訓練損失還是別的量?", "not “$\\Longrightarrow L_{\\mathrm{test}}$ is small”. Ask two things first: does my problem meet the assumptions? Is it the training loss that is guaranteed, or something else?"), size=22)
    banner(g, T("KP15 的兩個結果只是來源列舉的許多收斂理論中的兩個", "KP15's two results are just two of the many convergence theories the source lists"))
    d.done(g, "條件式光譜", "A conditional spectrum", {0: ("KP15 的兩個結果只是來源列舉的許多收斂理論中的兩個。來源把其他工作分成三個方向。第一是啟用函式：二次啟用、ReLU、Leaky ReLU。", "KP15's two results are only two of the many convergence theories the source lists, which sorts the others into three directions. First, activations: quadratic, ReLU, Leaky ReLU."), 1: ("第二是網路架構：殘差連接、非獨立同分布的資料流。", "Second, architectures: residual connections and non-i.i.d. data streams."), 2: ("第三是幾何與界：臨界點的刻畫、交叉熵的隱式偏置。", "Third, geometry and bounds: characterizing critical points and the implicit bias of cross-entropy."), 3: ("", ""), 4: ("它們的共同形式與 KP15 一樣：假設成立，則訓練損失小。", "Their common form is KP15's: if the assumptions hold, the training loss is small."), 5: ("而不是測試損失很小。讀任何一個這類結果，先問兩件事：它的假設我的問題滿足嗎？它保證的是訓練損失還是別的量？", "Not that the test loss is small. Reading any such result, first ask: does my problem meet its assumptions, and is it the training loss that is guaranteed or something else?")})

    g = d.page("全章工具鏈:問題對工具", "The chapter's toolchain: problems and tools")
    rows = [["KP01", T("訓練是什麼問題?", "what problem is training?"), T("在參數空間搜尋;只用便宜的一階梯度", "search in parameter space with the cheap first-order gradient")], ["KP02–05", T("一步該走多遠、往哪走?", "how far, which way per step?"), T("梯度下降與學習率、停止準則、線搜索、動量", "gradient descent and learning rate, stopping, line search, momentum")], ["KP06", T("梯度怎麼算?", "how is the gradient computed?"), T("反向傳播(鏈式法則,成本 $O(W)$)", "backpropagation (chain rule, cost $O(W)$)")], ["KP07–09", T("算得起嗎?", "can we afford it?"), T("SGD、小批次與 epoch;固定步長帶子、遞減步長", "SGD, minibatches, epochs; fixed-step band, decaying step")], ["KP10–11", T("步長能逐座標自動調嗎?", "can the step adapt per coordinate?"), T("AdaGrad、RMSProp、Adam(偏差修正)", "AdaGrad, RMSProp, Adam (bias correction)")], ["KP12–14", T("只看這一點的損失夠嗎?", "is the loss at one point enough?"), T("平坦極小值的動機;SAM、高效與自適應 SAM", "flat minima; SAM, efficient and adaptive SAM")], ["KP15", T("非凸為什麼還練得動?", "why does non-convex still train?"), T("過參數化下有條件的收斂保證", "conditional guarantees under over-parameterization")], ["KP16", T("練好之後呢?", "and after training?"), T("驗證與測試證據", "validation and test evidence")]]
    cols = [BLUE, BLUE, BLUE, ORG, GRN, PUR, RED, RED]; lights = [BLUEL, BLUEL, BLUEL, ORGL, GRNL, PURL, REDL, REDL]
    g.beat = 1
    table(g, 50, 108, [130, 410, 640], [T("知識點", "KPs"), T("問題", "question"), T("工具", "tool")], rows, rh=54, fs=18, hfs=16, styles=[(l, c) for c, l in zip(cols, lights)], beats=[1, 1, 1, 2, 2, 3, 3, 3], gap=4)
    banner(g, T("每個工具都有代價與適用條件;沒有任何一個回答所有問題", "every tool has a cost and conditions; none answers every question"))
    d.done(g, "工具鏈", "The toolchain", {0: ("整章可以看成一串問題，每個知識點回答其中一個。", "The whole chapter is a chain of questions, each unit answering one."), 1: ("前三列：訓練是什麼問題、一步該走多遠、梯度怎麼算，對應 KP01 到 KP06。", "The first rows: what training is, how far per step, how the gradient is computed, KP01 to KP06."), 2: ("接著：算得起嗎、步長能逐座標自動調嗎，對應 KP07 到 KP11。", "Then: can we afford it and can the step adapt per coordinate, KP07 to KP11."), 3: ("最後：只看一點的損失夠嗎、非凸為什麼還練得動、練好之後呢，對應 KP12 到 KP16。這條鏈上每個工具都有代價與適用條件，沒有任何一個回答所有問題。", "Finally: is one point's loss enough, why non-convex still trains, and after training, KP12 to KP16. Every tool on the chain has a cost and conditions; none answers every question.")})

    g = d.page("已回答與仍開放", "What is answered and what is still open")
    g.beat = 1
    box(g, 50, 126, 570, 360, GRN, GRNL, T("已回答", "answered"), T("1 梯度可以便宜地算出來(反向傳播)\n2 明確條件下 GD 與 SGD 的行為可分析\n3 自適應步長與鄰域最壞損失是有推導的設計\n4 過參數化、隨機初始化、資料可分時,訓練損失可證明降到任意小", "1 gradients can be computed cheaply (backpropagation)\n2 GD and SGD behavior is analyzable under clear conditions\n3 adaptive steps and the worst neighborhood loss are derived designs\n4 under over-parameterization, random init and separable data, training loss provably goes to any small value"), ts=26, bs=20)
    g.beat = 2
    box(g, 660, 126, 570, 360, RED, REDL, T("仍開放", "still open"), T("1 為什麼會泛化:定理沒說走到哪個解\n2 平坦度是不是泛化的原因:曲率與參數化有關\n3 最佳化器怎麼比較:要受控實驗,沒有通用排行", "1 why it generalizes: the theorem does not say which solution is reached\n2 whether flatness causes generalization: curvature depends on parameterization\n3 how to compare optimizers: controlled experiments, no universal ranking"), ts=26, bs=20)
    g.beat = 3
    gold(g, 150, 520, 980, 100, T("練好之後,還要問:沒看過的資料上表現如何?", "after training we still ask: how does it do on unseen data?"), size=26)
    banner(g, T("本章工具把訓練做好;「為什麼會泛化」留給下一章", "this chapter's tools make training work; “why it generalizes” goes to the next"))
    d.done(g, "已回答與仍開放", "Answered and open", {0: ("已回答四件事：梯度可以便宜地算；GD 與 SGD 的行為在明確條件下可分析；自適應步長與鄰域最壞損失是有推導的設計；過參數化等條件下訓練損失可證明降到任意小。", "Four things are answered: gradients can be computed cheaply; GD and SGD behavior is analyzable under clear conditions; adaptive steps and the worst neighborhood loss are derived designs; under over-parameterization and similar conditions the training loss provably goes to any small value."), 1: ("仍開放三件事：為什麼會泛化，定理沒說走到哪一個解；平坦度是不是泛化的原因，曲率與參數化有關；最佳化器怎麼比較，要靠受控實驗。", "Three remain open: why it generalizes, since the theorem does not say which solution is reached; whether flatness causes generalization, since curvature depends on parameterization; how to compare optimizers, which needs controlled experiments."), 2: ("練好之後，還要問：沒看過的資料上表現如何？", "After training we still ask: how does it do on unseen data?"), 3: ("", "")})

    g = d.page("訓練、驗證、測試各回答不同問題", "Training, validation and test answer different questions")
    g.beat = 1
    box(g, 50, 126, 370, 240, BLUE, BLUEL, T("訓練集", "training set"), T("調參數\n訓練目標有沒有下降?", "tunes the parameters\ndid the training objective fall?"), ts=24, bs=22)
    g.beat = 2
    box(g, 455, 126, 370, 240, ORG, ORGL, T("驗證集", "validation set"), T("選模型與超參數\n受控比較下誰比較好?", "chooses models and hyperparameters\nwho wins in a controlled comparison?"), ts=24, bs=22)
    g.beat = 3
    box(g, 860, 126, 370, 240, GRN, GRNL, T("測試集", "test set"), T("最後才碰一次\n沒碰過的資料上表現如何?", "touched only once, at the end\nhow does it do on untouched data?"), ts=24, bs=22)
    g.beat = 4
    gold(g, 150, 410, 980, 100, T("用訓練損失選模型,會選到最會背資料的那一個", "choosing by training loss picks the model that memorizes best"), size=26, light=REDL, stroke=RED)
    banner(g, T("三個集合互斥", "the three sets are disjoint"))
    d.done(g, "三個集合", "Three sets", {0: ("三個集合各有一個工作。訓練集調參數，回答訓練目標有沒有下降。", "Each of the three sets has a job. The training set tunes the parameters and answers whether the training objective fell."), 1: ("驗證集選模型與超參數，回答在受控比較下誰比較好。", "The validation set chooses models and hyperparameters and answers who is better in a controlled comparison."), 2: ("測試集最後才碰一次，回答沒碰過的資料上表現如何。", "The test set is touched once at the end and answers how it performs on untouched data."), 3: ("如果用訓練損失選模型，會選到最會背資料的那一個。", "Choosing a model by training loss picks the one that memorizes best.")})

    g = d.page("可以重算的例子:多項式次數", "A recomputable example: the polynomial degree")
    g.beat = 1
    g.text(50, 108, 1180, 36, T("真實曲線 $\\sin(3x)$,雜訊標準差 $0.2$(最好的測試 MSE $=0.04$);15 筆訓練、15 筆驗證、2000 筆測試", "true curve $\\sin(3x)$, noise std $0.2$ (best test MSE $=0.04$); 15 training, 15 validation, 2000 test samples"), size=19, color=MUTED, bold=True, align="center", z=8)
    rows = []
    for dg, (a, b_, c_) in EXP.items():
        rows.append([f"$d={dg}$", f"{a:g}", (f"{b_:,.2f}" if b_ > 10 else f"{b_:g}"), (f"{c_:,.2f}" if c_ > 10 else f"{c_:g}")])
    table(g, 80, 150, [260, 270, 290, 290], [T("次數", "degree"), T("訓練 MSE", "training MSE"), T("驗證 MSE", "validation MSE"), T("測試 MSE", "test MSE")], rows, rh=70, fs=26, hfs=20, styles=[(GRNL, GRN), (REDL, RED), (GOLDL, GOLD)], beats=[1, 2, 3], gap=8)
    g.beat = 4
    box(g, 50, 440, 570, 160, RED, REDL, T("依訓練損失選", "choose by training loss"), T("選到 $d=14$(測試 $164.76$)\n訓練 MSE 一路降,$d\\ge14$ 時是 $0$", "picks $d=14$ (test $164.76$)\ntraining MSE keeps falling, $0$ for $d\\ge14$"), ts=20, bs=20)
    g.beat = 5
    box(g, 660, 440, 570, 160, GRN, GRNL, T("依驗證損失選", "choose by validation loss"), T("選到 $d=3$(測試 $0.0571$)\n接近雜訊下限 $0.04$", "picks $d=3$ (test $0.0571$)\nclose to the noise floor $0.04$"), ts=20, bs=20)
    banner(g, T("驗證的 $0.0533$ 比測試的 $0.0571$ 略低:從 25 個模型挑最低,略偏樂觀;最後要用沒碰過的測試集", "validation $0.0533$ < test $0.0571$: the best of 25 is optimistic; report on the test set"))
    d.done(g, "多項式例子", "The polynomial example", {0: ("可以自己重算的例子：真實曲線是三倍 x 的正弦，標籤雜訊標準差 0.2，所以最好的測試誤差是 0.04。用不同次數的多項式做最小平方。", "A recomputable example: the true curve is sine of 3x with label noise std 0.2, so the best test error is 0.04. Fit polynomials of different degrees by least squares."), 1: ("次數 3：訓練 0.026，驗證 0.0533，測試 0.0571，接近雜訊下限。", "Degree 3: training 0.026, validation 0.0533, test 0.0571, close to the noise floor."), 2: ("次數 14，參數剛好等於資料筆數：訓練 0，驗證 1072，測試 164.76，最糟。", "Degree 14, parameters equal to the number of samples: training 0, validation 1072, test 164.76, the worst."), 3: ("次數 24，參數比資料多：訓練 0，驗證 0.5197，測試 0.8247，比次數 14 好很多，但仍比次數 3 差。", "Degree 24, more parameters than data: training 0, validation 0.5197, test 0.8247, far better than degree 14 but still worse than degree 3."), 4: ("依訓練損失選會選到次數 14，測試 164.76。", "Choosing by training loss picks degree 14 with test 164.76."), 5: ("依驗證損失選到次數 3，測試 0.0571。驗證數字略偏樂觀，所以最後的數字要用沒碰過的測試集。", "Choosing by validation loss picks degree 3 with test 0.0571. The validation number is slightly optimistic, so report on the untouched test set.")})
    closing(d, [(RED, REDL, "「收斂定理證明了模型會預測得好」", "“a convergence theorem proves the model predicts well”", "定理保證訓練損失;測試表現要靠驗證與測試證據", "it guarantees training loss; test performance needs validation and test evidence"), (ORG, ORGL, "「驗證集的數字就是最終表現」", "“the validation number is the final performance”", "從多個候選挑最低略偏樂觀;最後用沒碰過的測試集", "the best of several is slightly optimistic; use the untouched test set"), (PUR, PURL, "「這章的最佳工具是 Adam 加 SAM」", "“this chapter's best tool is Adam plus SAM”", "每個工具回答不同的問題,都有條件;沒有通用最佳", "each tool answers a different question with conditions: no universal best")],
            ("訓練損失降得再低,模型也可能只是把訓練資料背起來。怎麼控制複雜度、避免過擬合?", "However low the training loss, a model may just memorize. How do we control complexity and avoid overfitting?"), ("下一章:正則化與過擬合", "next chapter: regularization and overfitting"))
    return d.pages
