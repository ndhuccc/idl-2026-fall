import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp09_nums import *

KP = "KP09"; N = 10
TEAL = "#0f766e"

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1:時間軸 ============
    g = new(1, T("KP09 架構與訓練共同推動復興", "KP09 Architecture and training drive the revival"))
    ev = [(1980, BLUE, T("局部表示", "local representation"), "Neocognitron"), (1986, ORG, T("多層學習", "multilayer learning"), T("誤差反傳", "backprop")), (1989, BLUE, T("可訓練卷積網", "trainable CNN"), T("郵遞區號", "zip codes")),
          (1998, BLUE, "LeNet-5", T("文件辨識", "documents")), (2006, ORG, T("逐層預訓練", "layer-wise pretraining"), "DBN"), (2012, GRN, T("資料 + 算力", "data + compute")), (2017, PUR, T("序列架構", "sequence architecture"), "Transformer")]
    ev[5] = (2012, GRN, T("資料 + 算力", "data + compute"), "AlexNet")
    g.line([(60, 300), (1220, 300)], GRAY, sw=7, z=2, arrow="end", head=22)
    for i, (yr, col, nm, sub) in enumerate(ev):
        g.beat = 1 if i < 3 else (2 if i < 5 else 3)
        x = 120 + i * 168
        g.ellipse(x, 300, 18, fill=col, stroke="#ffffff", sw=4, z=5); g.text(x - 60, 225, 120, 40, str(yr), size=26, color=col, bold=True, align="center", vcenter=True, z=8)
        g.rect(x - 78, 340, 156, 170, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(x - 74, 350, 148, 80, nm, size=19 if zh else 16, color=col, bold=True, align="center", z=8); g.text(x - 74, 440, 148, 60, sub, size=17, color=INK, align="center", z=8)
    g.beat = 3
    g.text(50, 540, 1180, 70, T("這些是不同「可行條件與方法」的標記，不是新法一出、舊法立即失效的直線進化", "These mark different feasible conditions and methods, not a straight line where each new method kills the old"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("沒有單一演算法「喚醒」深度學習", "No single algorithm “woke up” deep learning"))
    pages.append((g, T("復興的時間軸", "The revival timeline"), {
        0: T("1980、1986、1989、1998、2006、2012、2017 是選取的文獻節點，各自對應不同的問題：局部表示、多層學習、資料與算力、序列架構。", "1980, 1986, 1989, 1998, 2006, 2012 and 2017 are selected papers, each tied to a different question: local representation, multilayer learning, data and compute, sequence architecture."),
        1: T("1998 年的 LeNet-5 和 2006 年的逐層預訓練。", "LeNet-5 in 1998 and layer-wise pretraining in 2006."),
        2: T("2012 年的 AlexNet 與 2017 年的 Transformer。這個時間軸是不同可行條件與方法的標記。", "AlexNet in 2012 and the Transformer in 2017. The timeline marks different feasible conditions and methods."),
        3: T("沒有單一演算法喚醒深度學習：資料表示、可訓練性、算力與任務目標一起改變可行的方法。", "No single algorithm woke deep learning: data representation, trainability, compute and the task objective together change what is feasible.")}))

    # ============ 2:局部結構與權重共享 ============
    g = new(2, T("相鄰像素是局部圖案：同一個濾波器反覆使用", "Neighboring pixels form local patterns: reuse one filter"))
    g.beat = 1
    card(g, 50, 130, 560, 470, BLUE); head(g, 70, 136, 520, T(r"$4\times4$ 單通道輸入", r"$4\times4$ single-channel input"), BLUE, 22)
    cs = 100
    for r in range(4):
        for c in range(4): g.rect(120 + c * cs * 0.9, 220 + r * cs * 0.9, cs * 0.9 - 6, cs * 0.9 - 6, fill="#e2e8f0", stroke=GRAY, sw=2, radius=6, z=3)
    g.beat = 2
    for (r, c, col) in ((0, 0, ORG), (0, 1, GRN), (1, 2, PUR)):
        g.rect(120 + c * cs * 0.9 - 4, 220 + r * cs * 0.9 - 4, 2 * cs * 0.9 - 2, 2 * cs * 0.9 - 2, fill=f"rgba({ {ORG:'194,65,12', GRN:'21,128,61', PUR:'109,40,217'}[col] },0.12)", stroke=col, sw=5, radius=8, z=5)
    for (r, c, col) in ((0, 0, ORG), (0, 1, GRN), (1, 2, PUR)): g.ellipse(120 + c * cs * 0.9 + cs * 0.9 - 4, 220 + r * cs * 0.9 + cs * 0.9 - 4, 11, fill=col, stroke="#ffffff", sw=2, z=7)
    g.text(70, 560, 520, 34, T(r"$2\times2$ 濾波器滑動，共 $9$ 個位置", r"a $2\times2$ filter slides over $9$ positions"), size=20, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    card(g, 650, 130, 580, 470, ORG); head(g, 670, 136, 540, T("每個位置，用同樣的 4 個權重", "Same 4 weights at every position"), ORG, 22)
    for r in range(2):
        for c in range(2): g.rect(780 + c * 110, 230 + r * 110, 100, 100, fill=ORGL, stroke=ORG, sw=3, radius=10, z=3); g.text(780 + c * 110, 230 + r * 110, 100, 100, rf"$w_{{{r * 2 + c + 1}}}$", size=30, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(670, 470, 540, 110, T("局部連結 + 權重共享：\n位置換了，參數不用重學", "locality + weight sharing:\nnew position, no new parameters"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("輸出是平移等變，不是完全平移不變", "The output is translation-equivariant, not fully translation-invariant"))
    pages.append((g, T("局部連結與權重共享", "Locality and weight sharing"), {
        0: T("影像相鄰的像素，通常形成局部圖案。這一頁用一個 4 乘 4 的單通道輸入，是講義的補充例子。", "Neighboring image pixels usually form local patterns. This page uses a 4-by-4 single-channel input, a supplementary example."),
        1: T("一個 2 乘 2 的濾波器在輸入上滑動，stride 是 1、不補邊，共有 9 個位置。", "A 2-by-2 filter slides over the input with stride 1 and no padding, over 9 positions."),
        2: T("每個位置都用同樣的 4 個權重，這叫權重共享，加上局部連結，位置換了參數不必重學。", "Every position uses the same 4 weights, called weight sharing; with local connectivity, a new position needs no new parameters."),
        3: T("輸出是平移等變，不要說成完全平移不變。", "The output is translation-equivariant; do not call it fully translation-invariant.")}))

    # ============ 3:參數數目 ============
    g = new(3, T("同樣的輸出形狀，參數差 30 倍", "The same output shape, a 30-fold gap in parameters"))
    pill(g, 1090, 84, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 124, 1180, 110, GRAY); g.text(70, 128, 1140, 102, T(r"兩張 $3\times3$ 特徵圖 $\Rightarrow$ $3\times3\times2=18$ 個輸出；輸入 $4\times4=16$ 個像素", r"two $3\times3$ maps $\Rightarrow$ $3\times3\times2=18$ outputs; input $4\times4=16$ pixels"), size=26, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    U = 1.8
    g.text(60, 270, 280, 50, T("全連接", "fully connected"), size=24, color=RED, bold=True, vcenter=True, z=8); g.rect(340, 270, fc * U, 50, fill=REDL, stroke=RED, sw=3, radius=6, z=3)
    g.text(340 + fc * U + 14, 270, 260, 50, rf"${n_pix}\times{n_out}+{n_out}={fc}$", size=24, color=INK, bold=True, vcenter=True, z=8)
    g.beat = 3
    g.text(60, 380, 280, 50, T("卷積（共享）", "convolution (shared)"), size=24, color=GRN, bold=True, vcenter=True, z=8); g.rect(340, 380, conv * U, 50, fill=GRNL, stroke=GRN, sw=3, radius=6, z=3); star(g, 860, 405, 18, GOLD, z=6, stroke="#b45309", sw=2)
    g.text(340 + conv * U + 14, 380, 380, 50, rf"$2\times(2\times2+1)={conv}$", size=24, color=INK, bold=True, vcenter=True, z=8)
    g.text(60, 470, 1160, 110, T("式中的 $+1$ 是每個濾波器的偏置；這只比較「同樣輸出形狀的一層」\n不表示 CNN 整體一定更小、更準", "the $+1$ is each filter's bias; this compares one layer with the same output shape\nit does not mean a CNN is smaller or more accurate overall"), size=21, color=MUTED, bold=True, align="center", z=8)
    banner(g, T(rf"${fc}$ 對 ${conv}$：同一層，共享權重省下大量參數", rf"${fc}$ versus ${conv}$: sharing weights saves a lot in one layer"))
    pages.append((g, T("一層的參數數目", "Parameters of one layer"), {
        0: T("這是講義的補充，受控例：輸入是 4 乘 4 的單通道影像，2 乘 2 的核，stride 1、不補邊，輸出兩張 3 乘 3 的特徵圖，共 3 乘 3 乘 2 等於 18 個輸出。", "A supplementary controlled example: a 4-by-4 single-channel input, 2-by-2 kernels, stride 1, no padding, two 3-by-3 output maps, 3 times 3 times 2 equal to 18 outputs."),
        1: T("若每個輸出各和全部 16 個像素全連接，並各有偏置：16 乘 18 加 18，等於 306 個參數。", "If each output connects to all 16 pixels with its own bias: 16 times 18 plus 18, which is 306 parameters."),
        2: T("若兩個卷積核在各位置重用：2 乘 (2 乘 2 加 1) 等於 10 個參數。柱長依 306 與 10 繪製。這只比較同樣輸出形狀的一層，不表示 CNN 整體一定更小、更準。", "If two convolution kernels are reused at every position: 2 times (2 times 2 plus 1) equals 10 parameters. Bars are drawn to 306 and 10. This compares one layer with the same output shape and does not mean a CNN is smaller or more accurate overall."),
        3: T("兩個數目由程式計算並核對。", "Both counts were computed and checked by program.")}))

    # ============ 4:LeNet 別混 ============
    g = new(4, T("Neocognitron、早期卷積網、LeNet-5：三件不同的事", "Neocognitron, early CNN, LeNet-5: three things"))
    items = [(1980, BLUE, BLUEL, "Neocognitron", T("階層、局部接受野、\n相似特徵重用", "hierarchy, local receptive fields,\nreuse of similar features"), T("不是後來用反向傳播\n端到端訓練的 LeNet-5", "not the later LeNet-5\ntrained end to end by backprop")),
             (1989, ORG, ORGL, T("早期可訓練卷積網", "Early trainable CNN"), T("LeCun 等：\n郵遞區號辨識", "LeCun et al.:\nZIP-code recognition"), T("「LeNet 1989」\n不等於 LeNet-5", "“LeNet 1989”\nis not LeNet-5")),
             (1998, GRN, GRNL, "LeNet-5", T("《Gradient-Based Learning\nApplied to Document Recognition》", "“Gradient-Based Learning\nApplied to Document Recognition”"), T("具體的架構與論文", "a specific architecture\nand paper"))]
    for i, (yr, col, light, nm, sub, lim) in enumerate(items):
        g.beat = i + 1
        x = 50 + i * 400
        g.rect(x, 130, 380, 470, fill=light, stroke=col, sw=3, radius=18, z=3)
        g.text(x, 140, 380, 60, str(yr), size=36, color=col, bold=True, align="center", z=8); g.text(x + 10, 205, 360, 50, nm, size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(x + 20, 280, 340, 110, sub, size=18, color=INK, bold=True, align="center", z=8)
        g.ellipse(x + 40, 168, 10, fill=col, z=6)
        g.rect(x + 30, 420, 320, 140, fill=CARD, stroke=col, sw=2, radius=12, z=4, dash=True); g.text(x + 40, 430, 300, 120, lim, size=20, color=col, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("年份與型號要分開：不是同一個模型、同一篇論文", "Keep years and models apart: not one model, not one paper"))
    pages.append((g, T("三個不同的里程碑", "Three different milestones"), {
        0: T("1980 年 Fukushima 的 Neocognitron，用階層、局部接受野及相似特徵的重用處理位置變化；它不是後來用反向傳播端到端訓練的 LeNet-5。", "Fukushima's 1980 Neocognitron used hierarchy, local receptive fields and reuse of similar features to handle position changes; it is not the later LeNet-5 trained end to end by backpropagation."),
        1: T("LeCun 等人 1989 年的郵遞區號辨識，展示了早期可訓練的卷積網。", "LeCun et al.'s 1989 ZIP-code recognition showed an early trainable convolutional network."),
        2: T("1998 年的《Gradient-Based Learning Applied to Document Recognition》描述的是具體架構 LeNet-5。所以 LeNet 1989 與 LeNet-5 1998 不能當成同一型號和同一論文。", "The 1998 paper “Gradient-Based Learning Applied to Document Recognition” describes the specific architecture LeNet-5. So “LeNet 1989” and “LeNet-5 1998” are not the same model or paper."),
        3: T("這是講義更正來源混淆的地方。", "This is where the note corrects a confusion in the source.")}))

    # ============ 5:可訓練到可擴充 ============
    g = new(5, T("1986 年之後，深網還要翻過好幾道坎", "After 1986, deep nets still had hurdles to clear"))
    g.beat = 1
    hurdles = [T("梯度傳遞", "gradient flow"), T("初始化", "initialization"), T("資料規模", "data scale"), T("硬體", "hardware")]
    g.line([(80, 520), (1200, 520)], GRAY, sw=5, z=2)
    for i, nm in enumerate(hurdles):
        x = 150 + i * 250
        g.poly([(x, 520), (x + 50, 420), (x + 100, 520)], fill=REDL, stroke=RED, sw=4, z=3); g.text(x - 40, 530, 180, 36, nm, size=20, color=RED, bold=True, align="center", z=8)
    g.beat = 2
    card(g, 50, 130, 570, 230, ORG); head(g, 70, 136, 530, T("2006：兩篇不同的 Hinton 相關論文", "2006: two different Hinton papers"), ORG, 20)
    g.text(70, 190, 530, 160, T("Hinton–Osindero–Teh：Deep Belief Nets，逐層貪婪學習\nHinton–Salakhutdinov：深層自編碼器降維\n（來源把兩篇混在一起，不可合成一篇）", "Hinton–Osindero–Teh: Deep Belief Nets, greedy layer-wise\nHinton–Salakhutdinov: deep autoencoders for reduction\n(the source merges them; they are two papers)"), size=18, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    card(g, 660, 130, 570, 230, GRN); head(g, 680, 136, 530, T("2012：AlexNet 在 ImageNet 突破", "2012: AlexNet breaks through on ImageNet"), GRN, 20)
    g.text(680, 190, 530, 160, T("大型標註資料 + GPU + ReLU\n+ 資料增強 + dropout\n的組合", "large labeled data + GPUs + ReLU\n+ data augmentation + dropout\nin combination"), size=20, color=INK, bold=True, align="center", z=8)
    g.text(50, 575, 1180, 50, T("1990 年代到 2006 年並非所有神經網路研究都「冬眠」：CNN、語音與循環網路持續進行", "The years from the 1990s to 2006 were not a total hibernation: CNN, speech and recurrent work continued"), size=18, color=MUTED, bold=True, align="center", z=8) if False else None
    banner(g, T("反向傳播不等於立刻好訓練：條件一起到位才行", "Backprop did not make deep nets instantly easy to train: conditions had to align"))
    pages.append((g, T("可訓練到可擴充", "From trainable to scalable"), {
        0: T("1986 年的反向傳播里程碑，並沒有讓每個深網立刻好訓練。深度增加後，梯度傳遞、初始化、資料規模與硬體，都會成為障礙。同時核方法與 SVM 在某些任務與資料條件下仍有價值。", "The 1986 back-propagation milestone did not make every deep network easy to train at once. With depth, gradient flow, initialization, data scale and hardware became obstacles, while kernel methods and SVMs stayed valuable in some settings."),
        1: T("2006 年有兩篇需分開的 Hinton 相關論文：Hinton、Osindero、Teh 的 Deep Belief Nets 提出逐層貪婪學習法；Hinton 與 Salakhutdinov 研究用深層自編碼器降維。來源把作者混在一起，兩者都是復興脈絡的一部分，但不可合成一篇。", "In 2006 there are two separate Hinton-related papers: Hinton, Osindero and Teh's Deep Belief Nets proposed greedy layer-wise learning; Hinton and Salakhutdinov studied deep autoencoders for dimensionality reduction. The source mixes the authors; both belong to the revival, but are not one paper."),
        2: T("2012 年 Krizhevsky、Sutskever、Hinton 的 AlexNet，以大型標註資料、GPU、ReLU、資料增強與 dropout 的組合，在 ImageNet 挑戰得到突破。", "In 2012 Krizhevsky, Sutskever and Hinton's AlexNet broke through on ImageNet by combining large labeled data, GPUs, ReLU, data augmentation and dropout."),
        3: T("把 1990 年代到 2006 年寫成所有神經網路研究完全冬眠，會掩蓋 CNN、語音與循環網路等持續的工作。", "Describing the 1990s to 2006 as total hibernation hides continuing work on CNNs, speech and recurrent networks.")}))

    # ============ 6:序列 ============
    g = new(6, T("序列：遞迴一格一格傳，注意力一次看全部", "Sequences: recurrence vs attention"))
    g.beat = 1
    card(g, 50, 130, 570, 470, BLUE); head(g, 70, 136, 530, T("RNN / LSTM：遞迴狀態", "RNN / LSTM: a recurrent state"), BLUE, 22)
    for i in range(4):
        x = 100 + i * 120
        g.rect(x, 300, 90, 90, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(x, 300, 90, 90, rf"$h_{i + 1}$", size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
        if i < 3: g.line([(x + 94, 345), (x + 118, 345)], BLUE, sw=5, arrow="end", head=14, z=5)
        g.line([(x + 45, 460), (x + 45, 396)], GRAY, sw=3, arrow="end", head=12, z=4); g.text(x, 468, 90, 36, rf"$x_{i + 1}$", size=22, color=MUTED, align="center", z=8)
    g.beat = 2
    card(g, 660, 130, 570, 470, ORG); head(g, 680, 136, 530, T("注意力：各位置之間取資訊", "Attention: pick information across positions"), ORG, 22)
    for i in range(4): g.ellipse(740 + i * 120, 500, 24, fill=ORGL, stroke=ORG, sw=3, z=4); g.ellipse(740 + i * 120, 260, 24, fill=ORGL, stroke=ORG, sw=3, z=4)
    for i in range(4):
        for j in range(4): g.line([(740 + i * 120, 480), (740 + j * 120, 284)], "#fdba74", sw=2, z=3)
    g.beat = 3
    g.text(50, 600, 1180, 30, "", size=16, z=1)
    g.rect(660, 530, 570, 60, fill=PURL, stroke=PUR, sw=3, radius=10, z=3); g.text(660, 530, 570, 60, T("2017 Transformer：以注意力替代遞迴", "2017 Transformer: attention replaces recurrence"), size=20, color=PUR, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("BERT：編碼器式預訓練；GPT：自迴歸生成（2018）", "BERT: encoder-style pretraining; GPT: autoregressive generation (2018)"))
    pages.append((g, T("序列與注意力", "Sequences and attention"), {
        0: T("RNN 與 LSTM 以遞迴狀態處理序列：一格一格往下傳。", "RNNs and LSTMs process sequences with a recurrent state passed along step by step."),
        1: T("注意力可以在序列各位置之間選取資訊。但 2014 到 2015 年帶注意力的編碼器與解碼器仍可含 RNN，不能說注意力當時已消除遞迴計算。", "Attention can pick information across positions, but the 2014 to 2015 encoder-decoders with attention could still contain RNNs, so attention had not yet removed recurrence."),
        2: T("2017 年 Transformer 的論文才明確提出以注意力替代其序列轉換架構中的遞迴，改善訓練平行化。", "The 2017 Transformer paper explicitly replaced recurrence with attention in its sequence-transduction architecture, improving training parallelism."),
        3: T("2018 年 BERT 代表編碼器式預訓練，GPT 代表自迴歸生成路線。不同架構對應不同輸入與輸出需求。圖是示意。", "In 2018 BERT represents encoder-style pretraining and GPT the autoregressive generation route. Different architectures serve different input and output needs. The figure is a sketch.")}))

    # ============ 7:生成 ============
    g = new(7, T("生成模型：三種「做出像資料的樣本」的方法", "Generative models: three ways to sample"))
    g.beat = 1
    gens = [(BLUE, BLUEL, "VAE", T("2013 預印本 / 2014 會議", "2013 preprint / 2014 conf."), T("潛變量 + 變分目標", "latent variables + variational objective")), (ORG, ORGL, "GAN", "2014", T("生成器 對抗 判別器", "generator vs discriminator")), (GRN, GRNL, T("擴散式", "Diffusion"), T("2015 起；DDPM 2020", "since 2015; DDPM 2020"), T("學逐步逆向生成", "learn step-by-step reverse generation"))]
    for i, (col, light, nm, yr, sub) in enumerate(gens):
        g.beat = i + 1
        x = 50 + i * 400
        card(g, x, 130, 380, 470, col); head(g, x + 10, 136, 360, nm, col, 28); g.text(x + 10, 188, 360, 34, yr, size=18, color=MUTED, bold=True, align="center", z=8)
        if i == 0:
            g.rect(x + 40, 290, 80, 110, fill=light, stroke=col, sw=3, radius=10, z=3); g.ellipse(x + 190, 345, 36, fill=PURL, stroke=PUR, sw=3, z=4); g.rect(x + 260, 290, 80, 110, fill=light, stroke=col, sw=3, radius=10, z=3)
            g.line([(x + 124, 345), (x + 150, 345)], col, sw=4, arrow="end", head=12, z=4); g.line([(x + 230, 345), (x + 256, 345)], col, sw=4, arrow="end", head=12, z=4)
        if i == 1:
            g.rect(x + 40, 290, 120, 110, fill=light, stroke=col, sw=3, radius=10, z=3); g.rect(x + 220, 290, 120, 110, fill=REDL, stroke=RED, sw=3, radius=10, z=3); g.line([(x + 164, 330), (x + 216, 330)], col, sw=4, arrow="end", head=12, z=4); g.line([(x + 216, 370), (x + 164, 370)], RED, sw=4, arrow="end", head=12, z=4)
        if i == 2:
            for k in range(4):
                g.rect(x + 30 + k * 85, 300, 70, 90, fill=light, stroke=col, sw=3, radius=8, z=3, fillOpacity=0.35 + k * 0.2)
                if k < 3: g.line([(x + 102 + k * 85, 345), (x + 112 + k * 85, 345)], col, sw=3, arrow="end", head=8, z=4)
        g.text(x + 20, 440, 340, 130, sub, size=21, color=INK, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("樣本品質與成本取決於實作與資料：不能說哪一類全面勝出", "Quality and cost depend on implementation and data: no class wins across the board"))
    pages.append((g, T("生成模型", "Generative models"), {
        0: T("生成模型回答：如何產生像資料的樣本。VAE 用潛變量與變分目標；Kingma 與 Welling 的預印本是 2013，會議版本是 2014。", "Generative models answer how to produce data-like samples. VAEs use latent variables and a variational objective; Kingma and Welling's preprint is 2013 and the conference version 2014."),
        1: T("GAN 用生成器與判別器對抗，2014 年。", "GANs use an adversarial generator and discriminator, 2014."),
        2: T("擴散式模型學逐步逆向生成。Sohl-Dickstein 等人的擴散生成論文是 2015 年，Ho 等人 2020 的 DDPM 是重要改進，並非擴散模型首次登場。", "Diffusion models learn step-by-step reverse generation. Sohl-Dickstein et al.'s diffusion paper is from 2015; Ho et al.'s 2020 DDPM was an important improvement, not the first appearance."),
        3: T("三種方法的樣本品質、涵蓋度、訓練及取樣成本，都取決於具體實作與資料，不能宣稱一類全面勝過另一類。圖是示意。", "Sample quality, coverage, training and sampling cost depend on implementation and data; no class can be claimed better across the board. The figures are sketches.")}))

    # ============ 8:偏好 ============
    g = new(8, T("偏好對齊：RLHF 與 DPO 走不同的路", "Preference alignment: RLHF and DPO routes"))
    g.beat = 1
    card(g, 50, 130, 1180, 200, BLUE); head(g, 70, 136, 1140, T("典型 RLHF", "Typical RLHF"), BLUE, 24)
    for i, (nm, col) in enumerate(((T("偏好資料", "preference data"), BLUE), (T("擬合獎勵模型", "fit a reward model"), ORG), (T("最佳化策略", "optimize the policy"), GRN))):
        x = 100 + i * 360
        g.rect(x, 200, 280, 100, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(x, 200, 280, 100, nm, size=24, color=col, bold=True, align="center", vcenter=True, z=8)
        if i < 2: arrow(g, (x + 288, 250), (x + 352, 250), GRAY, sw=5, head=16)
    g.beat = 2
    card(g, 50, 360, 1180, 200, ORG); head(g, 70, 366, 1140, T("DPO（2023）", "DPO (2023)"), ORG, 24)
    for i, (nm, col) in enumerate(((T("偏好資料", "preference data"), BLUE), (T("直接的偏好最佳化目標", "a direct preference objective"), PUR))):
        x, w_ = (100, 380) if i == 0 else (620, 330)
        g.rect(x, 430, w_, 100, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(x, 430, w_, 100, nm, size=24 if i == 0 else 21, color=col, bold=True, align="center", vcenter=True, z=8)
    arrow(g, (488, 480), (612, 480), GRAY, sw=5, head=16)
    g.rect(980, 430, 230, 100, fill=REDL, stroke=RED, sw=3, radius=14, z=3, dash=True); g.text(985, 436, 220, 90, T("省去另外擬合的\n顯式獎勵模型", "skips a separately fitted\nexplicit reward model"), size=17, color=RED, bold=True, align="center", z=8)
    g.beat = 3
    banner(g, T("「無額外獎勵模型」不等於「無人類偏好」", "“No extra reward model” does not mean “no human preferences”"))
    pages.append((g, T("偏好對齊", "Preference alignment"), {
        0: T("偏好對齊是另一個問題。典型 RLHF 先估計偏好獎勵模型，再最佳化策略。不要說任何 RLHF 都有同一套步驟。", "Preference alignment is a different problem. Typical RLHF first estimates a preference reward model, then optimizes the policy. Do not claim every RLHF uses the same steps."),
        1: T("2023 年 DPO 的原論文，提供不需另外擬合顯式獎勵模型的直接偏好最佳化目標，但仍依賴偏好資料與參考策略條件。", "The 2023 DPO paper gives a direct preference-optimization objective that avoids a separately fitted explicit reward model, but still depends on preference data and reference-policy conditions."),
        2: T("不要把「無額外獎勵模型」說成「無人類偏好」。", "Do not turn “no extra reward model” into “no human preferences”."),
        3: T("講義的 4 題自我檢核留在講義。", "The note's 4 self-check questions stay in the note.")}))

    # ============ 9:地圖 ============
    g = new(9, T("先看資料結構與目標，方法只是候選", "Data and objective first; methods are candidates"))
    rows = [(T("影像", "Images"), BLUE, "CNN"), (T("序列", "Sequences"), ORG, "RNN / LSTM / Transformer"), (T("圖資料", "Graphs"), GRN, "ChebNet / GCN / GAT"), (T("生成樣本", "Generating samples"), PUR, "VAE / GAN / " + T("擴散", "diffusion")), (T("人類偏好", "Human preferences"), RED, "RLHF / DPO")]
    for i, (nm, col, meth) in enumerate(rows):
        g.beat = 1 if i < 2 else (2 if i < 4 else 3)
        y = 130 + i * 98
        g.rect(50, y, 360, 78, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(50, y, 360, 78, nm, size=26, color=col, bold=True, align="center", vcenter=True, z=8)
        g.line([(416, y + 39), (520, y + 39)], col, sw=5, arrow="end", head=16, z=5)
        g.rect(526, y, 700, 78, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(526, y, 700, 78, meth, size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    banner(g, T("箭頭指向候選，不是一對一：CNN、GNN、DPO 是不同軸上的名字", "Arrows point to candidates, not one-to-one: CNN, GNN, DPO live on different axes"))
    pages.append((g, T("資料結構與方法", "Data structure and methods"), {
        0: T("影像、序列、圖、生成、偏好，形成不同的問題入口。影像對應 CNN；序列對應 RNN、LSTM 與 Transformer。", "Images, sequences, graphs, generation and preferences are different entry points. Images suggest CNNs; sequences suggest RNNs, LSTMs and Transformers."),
        1: T("圖資料有 ChebNet、GCN、GAT；生成樣本有 VAE、GAN 與擴散式模型。", "Graphs have ChebNet, GCN and GAT; generation has VAEs, GANs and diffusion models."),
        2: T("人類偏好則看 RLHF 與 DPO。常見迷思：CNN、Transformer、GNN、生成模型、DPO 不是一列互相取代的版本；有些是輸入結構，有些是模型架構，有些是學習目標或訓練程序，先問問題，再比較條件。", "Human preferences point to RLHF and DPO. Misconception: CNN, Transformer, GNN, generative models and DPO are not versions replacing each other; some are input structures, some architectures, some objectives or procedures. Ask the question first, then compare conditions."),
        3: T("箭頭指向方法候選，沒有宣稱方法與任務一對一。", "The arrows point to method candidates and do not claim a one-to-one rule.")}))

    # ============ 10:界線 + 銜接 ============
    g = new(10, T("其他里程碑：每一個都有要保留的界線", "Other milestones: each comes with a limit to keep"))
    cards = [(BLUE, T("深度強化學習", "Deep RL"), T("DQN 用網路近似行動價值；AlphaGo 還結合樹搜尋，不等於單一 DQN", "DQN approximates action values; AlphaGo adds tree search, so it is not a single DQN")), (ORG, T("少樣本與自監督", "Few-shot and self-supervised"), T("從資料自身造學習訊號，不表示完全無目標；少樣本仍需合適的訓練與評估任務", "signals made from the data itself are not goal-free; few-shot still needs suitable training and evaluation tasks")),
             (GRN, T("圖資料", "Graph data"), T("GCN 的預印本與會議年份不同；鄰接與標籤條件要說清楚", "GCN's preprint and conference years differ; adjacency and label conditions must be clear")), (RED, T("部署與風險", "Deployment and risk"), T("聯邦學習不自動保證隱私；壓縮也可能改變品質，需分開量測", "federated learning does not guarantee privacy; compression can change quality; measure separately"))]
    for i, (col, ttl, sub) in enumerate(cards):
        g.beat = 1 if i < 2 else 2
        x = 50 + (i % 2) * 600; y = 130 + (i // 2) * 210
        g.rect(x, y, 580, 190, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.rect(x, y, 580, 50, fill=[BLUEL, ORGL, GRNL, REDL][i], radius=14, z=4); g.text(x + 10, y, 560, 50, ttl, size=22, color=col, bold=True, align="center", vcenter=True, z=8); g.text(x + 20, y + 60, 540, 120, sub, size=19, color=INK, bold=True, align="center", z=8)
        g.ellipse(x + 30, y + 25, 8, fill=col, z=6)
    g.beat = 3
    g.text(50, 560, 1180, 70, T("口訣：先看資料結構與目標，再看架構、訓練與成本", "Rule of thumb: start from the data and objective, then weigh architecture, training and cost"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T(r"下一站 KP10：把整章收束成選擇流程 $\to$", r"Next, KP10: turn the chapter into a decision map $\to$"))
    pages.append((g, T("界線與銜接", "Limits and next step"), {
        0: T("深度強化學習：DQN 用神經網路近似行動價值，AlphaGo 還結合樹搜尋，不能等同單一 DQN。少樣本與自監督：從資料自身造學習訊號，不表示完全無目標。", "Deep RL: DQN approximates action values with a network, while AlphaGo also uses tree search and cannot be equated with a single DQN. Few-shot and self-supervised: signals from the data itself do not mean no objective."),
        1: T("圖資料：GCN 的預印本與會議年份不同。部署與風險：聯邦學習不自動保證隱私，壓縮也可能改變品質，要分開量測。", "Graph data: GCN's preprint and conference years differ. Deployment and risk: federated learning does not guarantee privacy and compression can change quality, so measure each separately."),
        2: T("口訣：先看資料結構與目標，再看架構、訓練與成本。下一個單元把整章的任務、方法與歷史收束為選擇流程。", "Rule of thumb: start from data structure and objective, then weigh architecture, training and cost. The next unit turns the chapter's tasks, methods and history into a decision map."),
        3: T("講義的 4 題自我檢核留在講義。", "The note's 4 self-check questions stay in the note.")}))
    return pages
