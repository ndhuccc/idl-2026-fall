import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *
from lec01_kp02_nums import *

KP = "KP02"; N = 9

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)

    # ============ 1:蘋果、特徵、樣本 ============
    g = new(1, T("KP02 資料是向量，模型是映射", "KP02 Data are vectors, models are mappings"))
    g.beat = 1
    card(g, 50, 120, 520, 500, BLUE)
    head(g, 70, 126, 480, T("一顆蘋果 = 一筆樣本", "One apple = one sample"), BLUE)
    pill(g, 440, 132, 110, T("比喻 · 示意", "Analogy"), GRAY, size=16)
    g.ellipse(310, 380, 120, 112, fill="#dc2626", stroke="#991b1b", sw=4, z=3, gradientTo="#f87171", gradientType="radial")
    g.ellipse(262, 330, 22, 14, fill="rgba(255,255,255,0.55)", z=4)
    g.line([(310, 270), (318, 232)], "#78350f", sw=8, z=4); g.poly([(318, 240), (370, 214), (394, 250), (340, 262)], fill="#22c55e", stroke="#15803d", sw=3, z=4)
    g.text(150, 520, 320, 40, T("這是 1 筆樣本", "this is 1 sample"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    arrow(g, (580, 380), (650, 380), ORG)
    card(g, 660, 120, 570, 500, ORG)
    head(g, 680, 126, 530, T("每種量測 = 一個特徵", "Each measurement = one feature"), ORG)
    for i, (nm, v, col) in enumerate(((T("甜度", "sweetness"), 0.7, "#ec4899"), (T("脆度", "crunchiness"), 0.5, "#0ea5e9"), (T("大小", "size"), 0.85, "#84cc16"))):
        y = 220 + i * 120
        g.text(690, y, 150, 50, nm, size=26 if zh else 21, color=INK, bold=True, vcenter=True, z=8)
        g.rect(850, y + 10, 340, 30, fill="#e2e8f0", radius=15, z=3); g.rect(850, y + 10, 340 * v, 30, fill=col, radius=15, z=4)
        g.ellipse(850 + 340 * v, y + 25, 14, fill=CARD, stroke=col, sw=4, z=5)
    pill(g, 1100, 132, 110, T("示意", "Sketch"), GRAY, size=16)
    g.beat = 3
    banner(g, T("模型收到的數值，要有固定順序、相同意義", "A model needs values in a fixed order with the same meaning"))
    pages.append((g, T("資料是向量，模型是映射", "Data are vectors, models are mappings"), {
        0: T("KP01 留下一個問題：模型到底接收什麼？這個單元先固定資料怎麼表示。（蘋果是比喻，不是講義的資料。）", "KP01 left a question: what does a model receive? This unit fixes how data is represented. (The apple is an analogy, not the note's data.)"),
        1: T("一顆蘋果就是一筆樣本。", "One apple is one sample."),
        2: T("它可以有很多種量測：甜度、脆度、大小。每種量測叫一個特徵。長條只是示意，沒有實際數字。", "It can have many measurements: sweetness, crunchiness, size. Each measurement is a feature. The bars are a sketch with no real numbers."),
        3: T("模型接收的數值必須有固定順序和相同意義，不能一筆先放甜度、下一筆先放脆度。", "A model needs values in a fixed order with the same meaning; sweetness cannot come first in one sample and crunchiness first in the next.")}))

    # ============ 2:固定順序 ============
    g = new(2, T("每筆資料的座標順序要固定", "Every sample uses the same coordinate order"))
    def vec(x, color, labels, title, ok):
        card(g, x, 130, 340, 440, color)
        head(g, x + 10, 136, 320, title, color, 22)
        brackets(g, x + 70, 200, 200, 270, INK)
        for i, lb in enumerate(labels):
            y = 220 + i * 120
            g.rect(x + 90, y, 160, 90, fill=[PURL, BLUEL][i % 2] if ok else REDL, stroke=[PUR, BLUE][i % 2] if ok else RED, sw=3, radius=12, z=3)
            g.text(x + 90, y, 160, 90, lb, size=28 if zh else 20, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(x + 20, 480, 300, 80, "", size=20, z=8)
    g.beat = 1
    vec(70, BLUE, (T("甜度", "sweetness"), T("脆度", "crunchiness")), r"$\mathbf x_1$：" + T("第 1 筆", "sample 1"), True)
    vec(470, BLUE, (T("甜度", "sweetness"), T("脆度", "crunchiness")), r"$\mathbf x_2$：" + T("第 2 筆", "sample 2"), True)
    tick(g, 190, 520, 40, GRN, z=8); tick(g, 590, 520, 40, GRN, z=8)
    g.beat = 2
    vec(870, RED, (T("脆度", "crunchiness"), T("甜度", "sweetness")), r"$\mathbf x_3$：" + T("順序顛倒", "order swapped"), False)
    cross(g, 990, 518, 44, RED, z=8)
    g.beat = 3
    banner(g, T("第 1 個座標永遠是甜度，第 2 個永遠是脆度", "Coordinate 1 is always sweetness and coordinate 2 always crunchiness"))
    pages.append((g, T("座標順序要固定", "A fixed coordinate order"), {
        0: T("特徵向量把一筆資料的數值按固定順序排好。有 d 個特徵，向量就有 d 個座標。（甜度、脆度是示意，沒有數字。）", "A feature vector arranges one sample's values in a fixed order; with d features it has d coordinates. (Sweetness and crunchiness are a sketch with no numbers.)"),
        1: T("前兩筆樣本都先放甜度、再放脆度，順序一致。", "The first two samples both put sweetness first and crunchiness second."),
        2: T("第三筆把順序顛倒了：同一個座標位置，意思變成不一樣，模型就會被搞亂。", "The third swaps the order: the same position now means something different, which confuses the model."),
        3: T("每個座標位置的意義，必須固定。它們都是數值化的表示，不代表共用單位。", "Each coordinate position must keep its meaning. They share a numerical representation, not common units.")}))

    # ============ 3:編碼與圖像 ============
    g = new(3, T("圖像尺寸不是樣本數", "Image size is not the sample count"))
    g.beat = 1
    card(g, 50, 120, 400, 500, BLUE); head(g, 70, 126, 360, T(r"一張 $64\times64$ 灰階圖", r"One $64\times64$ grayscale image"), BLUE)
    for r in range(8):
        for c in range(8):
            v = int(40 + ((r * 7 + c * 13) % 9) * 24); g.rect(100 + c * 40, 200 + r * 40, 40, 40, fill=f"rgb({v},{v},{v})", stroke="#e2e8f0", sw=1, z=3)
    g.text(70, 530, 360, 70, T("每個小格是 1 個像素\n（這裡只畫 8×8 示意）", "each cell is 1 pixel\n(an 8×8 sketch here)"), size=18, color=MUTED, align="center", z=8)
    g.beat = 2
    arrow(g, (470, 380), (600, 380), ORG, sw=6, head=22); g.text(462, 300, 150, 60, T("攤平", "flatten"), size=26, color=ORG, bold=True, align="center", z=8)
    card(g, 620, 120, 610, 500, ORG); head(g, 640, 126, 570, T(rf"一個長向量：${4096}$ 個像素特徵", rf"One long vector: ${4096}$ pixel features"), ORG)
    for k in range(32):
        v = int(40 + ((k * 7) % 9) * 24); g.rect(900, 190 + k * 13, 80, 13, fill=f"rgb({v},{v},{v})", stroke="#e2e8f0", sw=1, z=3)
    brackets(g, 880, 182, 120, 432, INK)
    g.text(1020, 250, 190, 120, T(r"$d=4096$ 個座標", r"$d=4096$ coordinates"), size=24, color=ORG, bold=True, z=8)
    g.text(1020, 440, 190, 100, T("仍然只是 1 筆樣本", "still just 1 sample"), size=24, color=INK, bold=True, z=8)
    g.beat = 3
    banner(g, T("「紅色」「偏長」要先編碼成實數，才是座標", "Words like “red” must be encoded as real numbers first"))
    pages.append((g, T("圖像與編碼", "Images and encoding"), {
        0: T("再看一個例子：圖片。來源的 64×64 灰階圖有 64 乘 64 等於 4096 個像素。這裡只畫 8 乘 8 的示意。", "Another example: an image. The source's 64×64 grayscale image has 64 times 64 equals 4096 pixels; only an 8×8 sketch is drawn."),
        1: T("每個像素的灰階值是一個數字。", "Each pixel's grayscale level is a number."),
        2: T("把它們攤平，就得到 4096 個座標的向量。圖像尺寸不是資料筆數，這仍然只是一筆樣本。", "Flatten them into a vector with 4096 coordinates. Image size is not the sample count; this is still one sample."),
        3: T("另外，紅色、偏長只是描述，還不能直接當實數座標，要先說明如何編碼。", "Also, words like red or elongated are descriptions, not yet real coordinates; the encoding must be specified.")}))

    # ============ 4:資料矩陣 ============
    g = new(4, T("把樣本按直欄排好，就是資料矩陣", "Samples as columns form the data matrix"))
    g.beat = 1
    card(g, 90, 120, 1100, 100, GRAY)
    g.text(110, 124, 1060, 92, r"$\mathbf X=[\mathbf x_1\ \mathbf x_2\ \cdots\ \mathbf x_n]\in\mathbb R^{d\times n}$", size=44, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    X0, Y0, cw, ch = 330, 330, 150, 70
    for c in range(5):
        for r in range(2):
            x = X0 + c * cw; y = Y0 + r * ch
            if c < 3:
                g.rect(x, y, cw - 8, ch - 8, fill=BLUEL if r == 0 else ORGL, stroke=BLUE if r == 0 else ORG, sw=3, radius=8, z=3)
                g.text(x, y, cw - 8, ch - 8, f(X_first3[r][c]), size=30, color=INK, bold=True, align="center", vcenter=True, z=8)
            elif c == 3:
                g.text(x, y, cw - 8, ch - 8, r"$\cdots$", size=36, color=MUTED, align="center", vcenter=True, z=8)
    brackets(g, X0 - 20, Y0 - 8, 4 * cw - 10, 2 * ch + 8, INK, 5)
    g.line([(X0, 306), (X0 + 3 * cw - 8, 306)], BLUE, sw=4, z=4); g.text(X0, 262, 3 * cw, 36, T(r"直欄 = 1 筆樣本 $\mathbf x_i$（共 $n=12$ 筆）", r"column = 1 sample $\mathbf x_i$ ($n=12$ in total)"), size=22, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.line([(X0 - 36, Y0), (X0 - 36, Y0 + 2 * ch - 8)], ORG, sw=4, z=4); g.text(40, 350, 240, 90, T("橫列 = 1 種特徵\n" + r"（$d=2$）", "row = 1 feature\n" + r"($d=2$)"), size=22, color=ORG, bold=True, z=8)
    g.text(X0 + 3 * cw - 8, Y0 + 2 * ch + 20, 300, 60, T("省略號 = 另外 9 筆，不是新特徵", "the ellipsis = 9 more samples, not a new feature"), size=18, color=MUTED, z=8)
    g.beat = 3
    g.text(300, 520, 700, 90, T(r"完整矩陣是 $2\times12$：2 個橫列、12 個直欄", r"The full matrix is $2\times12$: 2 rows, 12 columns"), size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T(r"X 的形狀是 $d\times n$：特徵數在前，樣本數在後", r"The shape of $\mathbf X$ is $d\times n$: features first, samples second"))
    pages.append((g, T("資料矩陣", "The data matrix"), {
        0: T("固定了單筆表示，就能收集多筆。每筆樣本是 d 乘 1 的直欄；n 筆樣本併排成 d 乘 n 的資料矩陣。本講義稱水平為橫列，垂直為直欄。", "With one sample defined, collect several. Each sample is a d-by-1 column; n samples side by side form a d-by-n data matrix. Rows are horizontal, columns vertical."),
        1: T("來源互動有 12 個無單位的二維教學點。這裡展開前 3 筆：(1.5, 2.0)、(2.5, 2.5)、(2.0, 3.5)。（這是教學點，不是真實水果。）", "The source has 12 unitless two-dimensional teaching points. Here are the first 3: (1.5, 2.0), (2.5, 2.5), (2.0, 3.5). (Teaching points, not real fruit.)"),
        2: T("每個直欄是一筆樣本；橫列是同一種特徵。省略號代表另外 9 筆，不是新特徵。", "Each column is one sample; a row is one feature. The ellipsis stands for 9 more samples, not another feature."),
        3: T("完整矩陣有 2 個橫列、12 個直欄，形狀是 2 乘 12。", "The full matrix has 2 rows and 12 columns: 2 by 12.")}))

    # ============ 5:n×d(補充) ============
    g = new(5, T("軟體常把樣本放在橫列：先檢查形狀", "Software often puts samples in rows: check the shape"))
    pill(g, 1090, 100, 140, T("講義補充", "Added note"), GRAY, size=16)
    g.beat = 1
    card(g, 50, 130, 570, 470, BLUE); head(g, 70, 136, 530, T(r"本講義：$d\times n$（樣本在直欄）", r"This note: $d\times n$ (samples in columns)"), BLUE, 22)
    for c in range(8):
        for r in range(2): g.rect(100 + c * 56, 250 + r * 56, 50, 50, fill=BLUEL if c < 7 else "#e2e8f0", stroke=BLUE, sw=2, radius=6, z=3)
    brackets(g, 88, 242, 470, 124, INK); g.text(90, 380, 470, 60, T(r"每個直欄是 1 筆樣本", r"each column is 1 sample"), size=24, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.text(90, 440, 470, 64, r"$\mathbf X\in\mathbb R^{d\times n}$", size=32, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    arrow(g, (630, 360), (690, 360), ORG)
    card(g, 700, 130, 530, 470, ORG); head(g, 720, 136, 490, T(r"常見陣列：$n\times d$（樣本在橫列）", r"Common arrays: $n\times d$ (samples in rows)"), ORG, 22)
    for r in range(8):
        for c in range(2): g.rect(880 + c * 56, 200 + r * 38, 50, 32, fill=ORGL if r < 7 else "#e2e8f0", stroke=ORG, sw=2, radius=6, z=3)
    brackets(g, 868, 192, 130, 312, INK); g.text(720, 510, 490, 64, r"$\mathbf X^{\top}\in\mathbb R^{n\times d}$", size=32, color=INK, align="center", vcenter=True, z=8)
    g.text(1010, 300, 200, 100, T("每個橫列是\n1 筆樣本", "each row is\n1 sample"), size=22, color=ORG, bold=True, z=8)
    g.beat = 3
    banner(g, T("這不是數學錯誤，只是約定不同：相乘前先查形狀", "Not a math error, just a different convention: check shapes before multiplying"))
    pages.append((g, T("橫列還是直欄", "Rows or columns"), {
        0: T("這是講義的補充：軟體也常使用 n 乘 d、樣本按橫列的約定。", "This is a supplement: software also often uses n-by-d arrays with samples in rows."),
        1: T("本講義的約定：樣本在直欄，形狀是 d 乘 n。", "This note's convention: samples in columns, shape d by n."),
        2: T("常見陣列的約定剛好相反。這不是數學錯誤，但與本講義不同。", "Common arrays use the opposite convention. It is not a mistake, but it differs from this note."),
        3: T("寫矩陣乘法之前一定要先檢查形狀，不能只記第一個維度是樣本。", "Always check shapes before matrix multiplication rather than assuming the first dimension counts samples.")}))

    # ============ 6:映射 ============
    g = new(6, T("模型是映射：輸入向量進去，數值出來", "A model is a mapping: vectors in, numbers out"))
    g.beat = 1
    g.rect(90, 110, 1100, 70, fill=CARD, stroke=GRAY, sw=2, radius=14, z=1)
    g.text(110, 112, 1060, 66, r"$f_\theta:\mathbb R^d\longrightarrow\mathbb R^p,\qquad\mathbf x\longmapsto f_\theta(\mathbf x)$", size=34, color=INK, align="center", vcenter=True, z=8)
    card(g, 50, 230, 300, 360, BLUE); head(g, 60, 236, 280, T("輸入", "Input"), BLUE)
    brackets(g, 130, 300, 140, 200, INK)
    for i, nm in enumerate((r"$x_1$", r"$x_2$")):
        g.rect(148, 316 + i * 90, 104, 74, fill=BLUEL, stroke=BLUE, sw=3, radius=10, z=3); g.text(148, 316 + i * 90, 104, 74, nm, size=32, color=INK, align="center", vcenter=True, z=8)
    g.text(60, 520, 280, 60, T(r"$d=2$ 個座標", r"$d=2$ coordinates"), size=24, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    arrow(g, (360, 410), (440, 410)); card(g, 450, 230, 380, 360, ORG)
    head(g, 460, 236, 360, T(r"模型 $f_\theta$", r"Model $f_\theta$"), ORG)
    for i, nm in enumerate((r"$w_1$", r"$w_2$", r"$b$")):
        x = 480 + i * 110; g.ellipse(x + 40, 380, 40, fill=ORGL, stroke=ORG, sw=3, z=3); g.text(x, 340, 80, 80, nm, size=30, color=INK, bold=True, align="center", vcenter=True, z=8)
    g.text(460, 450, 360, 40, r"$\theta=(w_1,w_2,b)$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.text(460, 510, 360, 60, T("3 個參數", "3 parameters"), size=24, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    arrow(g, (840, 410), (920, 410), GRN); card(g, 930, 230, 300, 360, GRN); head(g, 940, 236, 280, T("輸出", "Output"), GRN)
    g.rect(980, 340, 200, 100, fill=GRNL, stroke=GRN, sw=3, radius=12, z=3); g.text(980, 340, 200, 100, r"$f_\theta(\mathbf x)$", size=34, color=INK, align="center", vcenter=True, z=8)
    g.text(940, 480, 280, 90, T(r"$p=1$ 個數值", r"$p=1$ number"), size=24, color=GRN, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("輸出 1 維，不等於只有 1 個參數", "A one-dimensional output does not mean one parameter"))
    pages.append((g, T("模型是映射", "A model is a mapping"), {
        0: T("模型是一族由參數 theta 決定的函數：把 d 維的輸入，映射成 p 維的輸出。選定 theta，才得到一個具體的函數。", "A model is a family of functions indexed by theta, mapping a d-dimensional input to a p-dimensional output. Fixing theta selects one concrete function."),
        1: T("來源例子取 d 等於 2：輸入 2 個數字。", "The source example takes d equal to 2: two input numbers."),
        2: T("模型裡有 2 個權重和 1 個偏移，共 3 個參數，合起來叫 theta。", "The model has 2 weights and 1 offset: 3 parameters, together called theta."),
        3: T("輸出 p 等於 1：1 個分數。p 是輸出維度，theta 是參數，兩者不同。p 也不一定等於分類的類別數。", "The output has p equal to 1: one score. p is output dimension, theta is the parameters; they differ, and p need not equal a class count.")}))

    # ============ 7:代入 ============
    g = new(7, T("固定輸入後，代入就得到輸出", "Fix the input, substitute, and read off the output"))
    g.beat = 1
    card(g, 90, 120, 1100, 150, GRAY)
    g.text(110, 126, 1060, 80, r"$f_\theta(\mathbf x)=w_1x_1+w_2x_2+b$", size=48, color=INK, align="center", vcenter=True, z=8)
    g.text(110, 206, 1060, 50, T(r"來源預設 $w_1=w_2=1,\ b=0$；輸入 $\mathbf x=(1.5,\,2.0)$", r"Source defaults $w_1=w_2=1,\ b=0$; input $\mathbf x=(1.5,\,2.0)$"), size=24, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.rect(90, 290, 530, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=12, z=2); g.text(100, 290, 510, 70, T(r"$w_1,w_2$：每個特徵的貢獻", r"$w_1,w_2$: each feature's contribution"), size=22, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.rect(660, 290, 530, 70, fill=ORGL, stroke=ORG, sw=3, radius=12, z=2); g.text(670, 290, 510, 70, T(r"$b$：所有分數的偏移", r"$b$: a shift of every score"), size=22, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.beat = 2
    u = 100; x0 = 120; y0 = 430
    g.rect(x0, y0, 1.5 * u, 70, fill=BLUEL, stroke=BLUE, sw=3, radius=6, z=3); g.rect(x0 + 1.5 * u, y0, 2.0 * u, 70, fill=ORGL, stroke=ORG, sw=3, radius=6, z=3)
    g.text(x0, y0, 1.5 * u, 70, r"$1\times1.5$", size=24, color=BLUE, bold=True, align="center", vcenter=True, z=8); g.text(x0 + 1.5 * u, y0, 2.0 * u, 70, r"$1\times2.0$", size=24, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.line([(x0, y0 + 100), (x0 + 3.5 * u, y0 + 100)], GRN, sw=5, z=4); g.line([(x0, y0 + 90), (x0, y0 + 110)], GRN, sw=5, z=4); g.line([(x0 + 3.5 * u, y0 + 90), (x0 + 3.5 * u, y0 + 110)], GRN, sw=5, z=4)
    g.text(x0, y0 + 114, 3.5 * u, 44, rf"$1.5+2.0+0={f(out[0])}$", size=28, color=GRN, bold=True, align="center", vcenter=True, z=8)
    pill(g, x0 + 3.5 * u + 20, y0 + 20, 170, T("加總 = 首尾相接", "sum = end to end"), GRAY, size=16)
    g.beat = 3
    arrow(g, (840, 465), (880, 465), GRN)
    g.rect(890, 420, 300, 90, fill=GRNL, stroke=GRN, sw=3, radius=12, z=3); g.text(890, 420, 300, 90, rf"$f_\theta(1.5,2.0)={f(out[0])}$", size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
    star(g, 1210, 420, 18, GOLD, z=6, stroke="#b45309", sw=2)
    banner(g, T("同一個輸入，參數不同，輸出就不同", "Same input, different parameters, different output"))
    pages.append((g, T("代入算輸出", "Substitute and compute"), {
        0: T("來源的加權和模型：f 等於 w1 乘 x1，加 w2 乘 x2，再加 b。權重調整各特徵的貢獻，b 調整所有分數的偏移。預設 w1、w2 都是 1，b 是 0。", "The source's weighted-sum model: f equals w1 times x1 plus w2 times x2 plus b. Weights set each feature's contribution; b shifts every score. Defaults: w1 and w2 are 1, b is 0."),
        1: T("對第一筆輸入 1.5 和 2.0，逐步代入：1 乘 1.5，加 1 乘 2.0，再加 0。加總畫成兩段首尾相接的長條。", "For the first input 1.5 and 2.0, substitute step by step: 1 times 1.5, plus 1 times 2.0, plus 0. The sum is drawn as two bars end to end."),
        2: T("結果是 3.5。所有數值已用程式重算。", "The result is 3.5. All values were recomputed by program."),
        3: T("固定輸入以後，是參數決定輸出的數值。", "For a fixed input, the parameters determine the output.")}))

    # ============ 8:12 點 ============
    g = new(8, T("同樣 12 個點：左邊 2 個座標，右邊 1 個分數", "Same 12 points: 2 coordinates in, 1 score out"))
    g.beat = 1
    card(g, 50, 120, 560, 500, BLUE); head(g, 70, 126, 520, T(r"輸入：$(x_1,x_2)$", r"Inputs: $(x_1,x_2)$"), BLUE)
    px0, py0, S = 110, 560, 38
    g.line([(px0, py0), (px0 + 10 * S + 10, py0)], MUTED, sw=3, z=3); g.line([(px0, py0), (px0, py0 - 10 * S - 10)], MUTED, sw=3, z=3)
    for t in (0, 5, 10):
        g.text(px0 + t * S - 20, py0 + 4, 40, 26, str(t), size=16, color=MUTED, align="center", z=8); g.text(px0 - 44, py0 - t * S - 14, 36, 26, str(t), size=16, color=MUTED, align="right", z=8)
    g.text(px0 + 10 * S + 14, py0 - 38, 60, 30, r"$x_1$", size=22, color=MUTED, z=8); g.text(px0 + 10, py0 - 10 * S - 12, 60, 30, r"$x_2$", size=22, color=MUTED, z=8)
    col = lambda v: "#2563eb" if v < 8 else ("#f59e0b" if v < 12 else "#dc2626")
    for (x, y), o in zip(pts, out): g.ellipse(px0 + x * S, py0 - y * S, 10, fill=col(o), stroke="#ffffff", sw=2, z=5)
    g.beat = 2
    card(g, 650, 120, 580, 500, GRN); head(g, 670, 126, 540, T("輸出：1 個分數（柱高 = 數值）", "Output: 1 score (bar height = value)"), GRN, 21)
    bx0, by0, bu = 700, 540, 20
    g.line([(bx0 - 10, by0), (bx0 + 12 * 41, by0)], MUTED, sw=3, z=3)
    for i, o in enumerate(out):
        g.rect(bx0 + i * 41, by0 - o * bu, 32, o * bu, fill=col(o), stroke="#ffffff", sw=1, radius=3, z=4)
        g.text(bx0 + i * 41 - 4, by0 + 4, 40, 24, str(i + 1), size=16, color=MUTED, align="center", z=8)
    g.text(bx0 - 4, by0 - out[0] * bu - 30, 44, 26, f(out[0]), size=16, color=INK, bold=True, align="center", z=8)
    mi = out.index(max(out)); g.text(bx0 + mi * 41 - 6, by0 - max(out) * bu - 30, 52, 26, f(max(out)), size=16, color=INK, bold=True, align="center", z=8); star(g, bx0 + mi * 41 + 16, by0 - max(out) * bu - 46, 12, GOLD, z=6)
    g.text(670, 566, 540, 40, T("樣本編號 1 到 12；最大輸出是 16", "sample 1 to 12; the maximum output is 16"), size=18, color=MUTED, align="center", z=8)
    g.beat = 3
    banner(g, T("加權和不一定是投影長度：$(1,1)$ 不是單位向量", "A weighted sum is not always a projection length: $(1,1)$ is not a unit vector"))
    pages.append((g, T("12 個點的輸出", "Outputs of the 12 points"), {
        0: T("左邊是來源互動的 12 個點，每個點有 2 個座標。顏色對應右邊輸出的大小：藍色小、橘色中、紅色大。", "On the left are the source's 12 points, each with 2 coordinates. Color matches the output on the right: blue small, orange medium, red large."),
        1: T("右邊是同一組參數下，12 個點的輸出。第一筆是 3.5，最大是 16，有兩筆輸出相同。", "On the right are the outputs of the 12 points under the same parameters. The first is 3.5, the maximum is 16, and two outputs coincide."),
        2: T("常見迷思：內積一定等於投影長度。只有方向是單位向量，內積才是該方向的帶號投影長度。這裡的方向 (1,1) 不是單位向量，所以輸出是加權和，不是投影長度。b 不為 0 時，這其實是仿射映射，來源用較寬的稱法叫線性模型。", "Common misconception: every inner product is a projection length. It is only so for a unit direction; (1,1) is not a unit vector, so the output is a weighted sum, not a projection length. With b nonzero this is an affine mapping; the source uses the broader term linear model."),
        3: T("所有 12 個輸出都由程式實算。", "All 12 outputs were computed by program.")}))

    # ============ 9:驗證了什麼 ============
    g = new(9, T("運算對了，不代表學會了", "Correct arithmetic does not mean it has learned"))
    g.beat = 1
    card(g, 50, 130, 560, 470, GRN); head(g, 70, 136, 520, T("已經檢查", "Checked"), GRN, 26)
    for i, s in enumerate((T("形狀：$d\\times n$ 與 $d$、$p$ 都對得上", "Shapes: $d\\times n$, $d$ and $p$ all agree"), T("運算：12 個輸出由程式重算", "Arithmetic: 12 outputs recomputed"), T(r"$\theta$ 是控制項指定的，不是訓練出來的", r"$\theta$ is set by controls, not trained"))):
        tick(g, 90, 220 + i * 120, 40, GRN, z=6) if i < 2 else (g.ellipse(110, 240 + i * 120, 20, fill=GOLD, z=5), g.text(90, 220 + i * 120, 40, 40, "!", size=26, color=INK, bold=True, align="center", vcenter=True, z=8))
        g.text(150, 205 + i * 120, 430, 80, s, size=22, color=INK, bold=True, vcenter=True, z=8)
    g.beat = 2
    card(g, 670, 130, 560, 470, RED); head(g, 690, 136, 520, T("還沒檢查", "Not checked"), RED, 26)
    g.text(700, 220, 500, 150, "?", size=110, color=RED, bold=True, align="center", vcenter=True, z=8)
    g.text(700, 380, 500, 90, T("預測品質：沒有訓練程序、\n也沒有和正確標籤比較", "Prediction quality: no fitting,\nno comparison with correct labels"), size=22, color=INK, bold=True, align="center", z=8)
    mascot(g, 1110, 440, 0.45, "confused")
    g.beat = 3
    banner(g, T(r"下一站 KP03：輸出是類別，還是連續值？$\to$", r"Next, KP03: is the output a category or a continuous value? $\to$"))
    pages.append((g, T("驗證了什麼", "What was verified"), {
        0: T("數值對得上公式，還不能代表模型已經學會任務。已經檢查的是：形狀對得上，12 個輸出也由程式重算過。", "Matching the arithmetic does not establish task performance. What has been checked: shapes agree and the 12 outputs were recomputed."),
        1: T("這裡的 theta 由互動控制項指定，沒有訓練程序，也沒有和正確標籤比較。所以預測品質沒有檢查。", "Here theta is set by interactive controls, with no fitting and no comparison to correct labels, so prediction quality is unchecked."),
        2: T("下一個單元會問：輸出是類別，還是連續值？", "The next unit asks: is the output a category or a continuous value?"),
        3: T("講義的 4 題小練習留在講義裡。", "The note's 4 practice questions stay in the note.")}))
    return pages
