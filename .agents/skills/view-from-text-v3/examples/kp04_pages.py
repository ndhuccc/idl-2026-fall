"""KP04 pages designed in HTML (view-from-text-v3, route A)."""
import os, sys, math
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp04_nums as K
W, H = 1160, 480
CA, CB = "#4f86bf", "#c4572e"; CAt, CBt = "#7fb0e6", "#e8896a"; LINE, GG, ACT = "#e0c85f", "#b7b0a5", "#f0a35a"
def row(x, y, w, h, role, a=.16): return card(x, y, w, h, role, a)
def arrow(x1, x2, y, col="#b7b0a5"): return f'<line x1="{x1}" y1="{y}" x2="{x2-8}" y2="{y}" stroke="{col}" stroke-width="3"/><polygon points="{x2},{y} {x2-12},{y-7} {x2-12},{y+7}" fill="{col}"/>'
def minus(v): return f"{v:+.1f}".replace("-", "−")
class Cube:
    def __init__(s, x0, y0, S): s.x0, s.y0, s.S = x0, y0, S
    def P(s, c): a, b, d = (int(ch) for ch in c); return (s.x0 + a * s.S + b * s.S * .55, s.y0 - d * s.S - b * s.S * .45)
    def edges(s):
        o = ""
        for c in K.CODES:
            for i in range(3):
                if c[i] == "0":
                    d = c[:i] + "1" + c[i + 1:]; (x1, y1), (x2, y2) = s.P(c), s.P(d); hid = "010" in (c, d)
                    o += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{GG}" stroke-width="{2 if hid else 3}"' + (' stroke-dasharray="7 6" opacity=".6"' if hid else "") + '/>'
        return o
    def lab(s, c, dx, dy, text, col="#f2ede4", size=16, bold=False, anchor="middle"):
        x, y = s.P(c); return f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{col}">{text}</text>'
LBL = {"000": (-30, 6, "end"), "100": (26, 6, "start"), "010": (-26, 6, "end"), "110": (26, 6, "start"), "001": (-26, 6, "end"), "101": (26, 6, "start"), "011": (-26, -14, "end"), "111": (26, -14, "start")}
def corner(cb, c, kind, r=17):
    x, y = cb.P(c)
    if kind == "A": return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{CA}"/><text x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">A</text>'
    if kind == "B": return f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}" rx="5" fill="{CB}"/><text x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">B</text>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#1c1712" stroke="{GG}" stroke-width="3" stroke-dasharray="5 4"/>'
def codelab(cb, c, text=None, col="#f2ede4", dy=0):
    dx, d0, an = LBL[c]; dx = dx * 1.5 if abs(dx) == 26 else dx; return cb.lab(c, dx, d0 + dy, text or c, col, 17, False, an)

# ------------------------------------------------ 1: target codes are scattered on the cube
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 700, 480, "blue", t("立方體的 8 個角：A 是 000、110、111", "The cube's 8 corners: A is 000, 110, 111"))
    cb = Cube(210, 400, 170); a = ""
    pts = " ".join(f"{cb.P(c)[0]:.1f},{cb.P(c)[1]:.1f}" for c in ("000", "100", "110", "010"))
    a += f'<polygon points="{pts}" fill="#e07a70" opacity=".14" stroke="#e07a70" stroke-width="3" stroke-dasharray="9 7"/>' + cb.edges()
    for c in K.CODES: a += corner(cb, c, "A" if c in K.TARGETS else "N" if c == "101" else "B") + codelab(cb, c)
    o += svg(0, 0, W, H, a)
    o += txt(500, 170, 190, t("101：沒有任何輸入會到達（空心圓）", "101: no input ever reaches it (hollow circle)"), 16, MUT, False, "left", 76)
    o += txt(16, 436, 668, t("紅色虛線框：底面 y₃ = 0，A 與 B 沿對角交錯", "Red dashed frame: the bottom face y₃ = 0, with A and B alternating along the diagonal"), 17, "#f08a80", True, "center", 26)
    o += cardH(716, 0, 444, 480, "teal", t("為什麼一個切面切不開", "Why one cut fails"))
    o += named(732, 46, 412, t("目標碼：標為 A 的角的編碼，這裡是 000、110、111。", "Target code: the code of a corner labelled A; here 000, 110, 111."), "term:" + t("目標碼", "Target code"), 16, TXT, 56)
    o += row(732, 112, 412, 100, "red", .18) + txt(740, 118, 396, t("底面 000 與 110 是 A，100 與 010 是 B：和 XOR 一樣交錯。", "On the bottom face 000, 110 are A and 100, 010 are B: alternating like XOR."), 17, TXT, False, "center", 84)
    o += row(732, 224, 412, 80, "grey", .12) + txt(740, 230, 396, t("最後一層只能用一個平面，把 8 個角分成兩邊。", "The last layer has one plane, which splits the 8 corners into two sides."), 17, TXT, False, "center", 64)
    o += card(732, 316, 412, 142, "yellow", .14) + txt(740, 322, 396, t("解法的想法：不要一刀分完。〔程式計算〕", "The idea: do not finish in one cut. (computed)"), 19, TXT, True, "center", 56)
    return o

# ------------------------------------------------ 2: three stages
def page2(lang):
    t = tr(lang); o = ""; a = ""
    o += cardH(0, 0, 1160, 250, "blue", t("三步走：編碼、偵測、合併", "Three steps: code, detect, merge"))
    xs = (20, 250, 520, 790, 1010)
    o += box(20, 62, 170, 150, "#5a544b", "#3a352d", 135, shadow=False, r=12) + txt(20, 70, 170, t("輸入 x", "input x"), 18, TXT, True, "center", 28) + txt(20, 108, 170, t("平面上的一點", "a point in the plane"), 16, TXT, False, "center", 50)
    o += box(250, 62, 220, 150, "#4f86bf", "#2f4f7a", 135, shadow=False, r=12) + txt(250, 70, 220, t("第一隱藏層", "1st hidden layer"), 17, TXT, True, "center", 28) + txt(250, 108, 220, t("三條線給三個位元<br>編碼 y = (y₁, y₂, y₃)", "three lines give three bits<br>code y = (y₁, y₂, y₃)"), 16, TXT, False, "center", 52)
    o += box(520, 62, 220, 150, "#d9822b", "#a8452b", 135, shadow=False, r=12) + txt(520, 70, 220, t("第二隱藏層", "2nd hidden layer"), 17, TXT, True, "center", 28) + txt(520, 108, 220, t("每個目標碼一個偵測器<br>b₀₀₀、b₁₁₀、b₁₁₁", "one detector per target code<br>b₀₀₀, b₁₁₀, b₁₁₁"), 16, TXT, False, "center", 52)
    o += box(790, 62, 190, 150, "#4f9a68", "#2c6140", 135, shadow=False, r=12) + txt(790, 70, 190, t("輸出層", "output layer"), 17, TXT, True, "center", 28) + txt(790, 108, 190, t("或合併<br>任一個啟動就判 A", "OR merge<br>A if any one fires"), 16, TXT, False, "center", 52)
    o += box(1010, 62, 130, 150, "#5a544b", "#3a352d", 135, shadow=False, r=12) + txt(1010, 112, 130, "A / B", 22, TXT, True, "center", 30)
    a += arrow(190, 250, 137) + arrow(470, 520, 137) + arrow(740, 790, 137) + arrow(980, 1010, 137); o += svg(0, 0, W, H, a)
    o += row(0, 262, 380, 108, "grey", .1) + row(396, 262, 380, 108, "grey", .1) + row(792, 262, 368, 108, "grey", .1)
    o += named(8, 270, 366, t("三層網路：兩個隱藏層加一個輸出層。", "Three-layer network: two hidden layers plus an output layer."), "term:" + t("三層網路", "Three-layer network"), 16, TXT, 60)
    o += named(404, 270, 366, t("偵測器：一個階梯單元，輸入碼剛好等於某個目標碼時輸出 1，否則輸出 0。", "Detector: a step unit that outputs 1 when the input code equals one target code, otherwise 0."), "term:" + t("偵測器", "Detector"), 16, TXT, 90)
    o += named(800, 270, 352, t("或合併（OR）：任何一個偵測器輸出 1 就判 A；全部是 0 就判 B。", "OR merge: A if any detector outputs 1; B if all are 0."), "term:" + t("或合併", "OR merge"), 16, TXT, 90)
    o += card(0, 380, 1160, 90, "yellow", .14) + txt(16, 392, 1128, t("上一個知識點的兩層網路只有第一層和輸出層，中間少了偵測器這一層。", "The previous unit's two-layer network had only the first layer and the output layer, with no detector layer between."), 19, TXT, True, "center", 60)
    return o

# ------------------------------------------------ 3: detect 110
def page3(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "blue", t("對 110 偵測器的分數 h₁₁₀ = y₁ + y₂ − y₃ − 1.5", "Scores of the 110 detector: h₁₁₀ = y₁ + y₂ − y₃ − 1.5"))
    cb = Cube(150, 400, 150); a = cb.edges()
    for c in K.CODES:
        v = K.h("110", c); x, y = cb.P(c); hot = c == "110"
        a += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="{"#4f9a68" if hot else "#5a544b"}"/>' + codelab(cb, c, f"{c}: {minus(v)}", "#f2ede4")
        a += cb.lab(c, 0, 5 if True else 0, "", "#f2ede4")
    o += svg(0, 0, W, H, a)
    for c in K.CODES:
        x, y = cb.P(c); o += txt(x - 18, y - 12, 36, c, 16, MUT, False, "center", 24) if False else ""
    o += txt(16, 430, 528, t("每個角旁邊是 h₁₁₀；只有 110 本身是正的（綠點）。〔程式計算〕", "Each corner shows h₁₁₀; only 110 is positive (green). (computed)"), 17, TXT, True, "center", 28)
    o += cardH(576, 0, 584, 480, "violet", t("每錯一個位元，分數少 1", "Every wrong bit costs 1 point"))
    o += card(592, 48, 552, 70, "violet", .22) + formula(592, 52, 552, r"h_t(\mathbf{y})=\sum_{i=1}^{3}s_iy_i-(n_1-0.5)", 24, 60)
    o += txt(592, 124, 552, t("t 是目標碼；n₁ 是 t 中 1 的個數；sᵢ = +1（tᵢ = 1）或 −1（tᵢ = 0）；m 是 y 與 t 不吻合的位元數。", "t: target code; n₁: number of 1s in t; sᵢ = +1 (tᵢ = 1) or −1 (tᵢ = 0); m: number of bits where y differs from t."), 16, TXT, False, "left", 76)
    o += card(592, 206, 552, 52, "teal", .2) + formula(592, 210, 552, r"\sum_i s_iy_i=n_1-m\ \Rightarrow\ h_t=0.5-m", 22, 44)
    for xx, lab, w_ in ((600, t("不吻合 m", "mismatches m"), 150), (760, t("分數 0.5 − m", "score 0.5 − m"), 170), (940, t("輸出 f(h)", "output f(h)"), 190)): o += txt(xx, 270, w_, lab, 16, MUT, True, "center", 24)
    for i in range(4):
        y0 = 296 + i * 38; v = 0.5 - i; o += row(592, y0, 552, 34, "green" if v > 0 else "grey", .22 if v > 0 else .1)
        o += txt(600, y0 + 4, 150, str(i), 17, TXT, False, "center", 26) + txt(760, y0 + 4, 170, minus(v), 17, TXT, True, "center", 26) + txt(940, y0 + 4, 190, "1" if v > 0 else "0", 17, TXT, True, "center", 26)
    o += txt(592, 452, 552, t("例：110 → m = 0，h = +0.5；010 → m = 1，h = −0.5。", "Example: 110 → m = 0, h = +0.5; 010 → m = 1, h = −0.5."), 16, MUT, False, "center", 24)
    return o

# ------------------------------------------------ 4: three detectors
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 520, 480, "violet", t("同一個做法，三個目標碼", "One recipe, three target codes"))
    texs = [r"h_{000}=-y_1-y_2-y_3+0.5", r"h_{110}=y_1+y_2-y_3-1.5", r"h_{111}=y_1+y_2+y_3-2.5"]
    notes = [t("000 沒有 1：偏置 +0.5", "000 has no 1: bias +0.5"), t("110 有兩個 1：偏置 −1.5", "110 has two 1s: bias −1.5"), t("111 有三個 1：偏置 −2.5", "111 has three 1s: bias −2.5")]
    for i, (tx, nt) in enumerate(zip(texs, notes)):
        y0 = 52 + i * 108; o += card(16, y0, 488, 98, "violet", .18) + formula(16, y0 + 6, 488, tx, 24, 52) + txt(16, y0 + 62, 488, nt, 17, TXT, False, "center", 28)
    o += card(16, 380, 488, 82, "yellow", .14) + txt(24, 386, 472, t("每個偵測器在自己的角得 +0.5。", "Each detector scores +0.5 at its own corner."), 19, TXT, True, "center", 30) + txt(24, 420, 472, t("〔程式計算〕", "(computed)"), 16, MUT, False, "center", 24)
    o += cardH(536, 0, 624, 480, "teal", t("八個編碼對三個偵測器的分數", "Scores of the three detectors on the eight codes"))
    for xx, lab, w_ in ((548, t("編碼 y", "code y"), 90), (648, "h₀₀₀", 140), (796, "h₁₁₀", 140), (944, "h₁₁₁", 140)): o += txt(xx, 46, w_, lab, 17, MUT, True, "center", 26)
    for i, c in enumerate(K.CODES):
        y0 = 78 + i * 46; o += row(548, y0, 600, 40, "blue" if c in K.TARGETS else "grey", .16 if c in K.TARGETS else .08)
        o += txt(548, y0 + 7, 90, c, 18, TXT, True, "center", 26)
        for j, tc in enumerate(K.TARGETS):
            v = K.h(tc, c); xx = 648 + j * 148
            if v > 0: o += row(xx + 10, y0 + 3, 120, 34, "green", .45)
            o += txt(xx, y0 + 7, 140, minus(v), 18, TXT, v > 0, "center", 26)
    return o

# ------------------------------------------------ 5: code 110 through the net
def page5(lang):
    t = tr(lang); o = ""; a = ""; y = "110"; hs = [K.h(tc, y) for tc in K.TARGETS]; bs = K.bits(y); qq = K.q(y)
    o += cardH(0, 0, 1160, 480, "orange", t("編碼 y = (1,1,0) 走過網路：只有 110 的偵測器啟動", "Code y = (1,1,0) through the network: only the 110 detector fires"))
    o += box(20, 190, 150, 100, "#5a544b", "#3a352d", 135, shadow=False, r=12) + txt(20, 200, 150, t("編碼 y", "code y"), 17, TXT, True, "center", 28) + formula(20, 236, 150, r"(1,1,0)", 24, 44)
    fm = [r"-1-1-0+0.5", r"1+1-0-1.5", r"1+1+0-2.5"]
    for i, tc in enumerate(K.TARGETS):
        y0 = 58 + i * 120; hot = bs[i] == 1
        a += f'<line x1="170" y1="240" x2="360" y2="{y0+52}" stroke="{ACT if hot else "#6b645a"}" stroke-width="{5 if hot else 2}"/><line x1="720" y1="{y0+52}" x2="800" y2="240" stroke="{ACT if hot else "#6b645a"}" stroke-width="{5 if hot else 2}"/>'
        o += card(360, y0, 360, 104, "green" if hot else "grey", .24 if hot else .1)
        o += txt(368, y0 + 6, 344, t(f"{tc} 的偵測器", f"detector {tc}"), 17, TXT, True, "center", 28) + formula(368, y0 + 32, 344, rf"h_{{{tc}}}={fm[i]}={minus(hs[i]).replace('−','-')}", 20, 36).replace("=-", "=-") + txt(368, y0 + 70, 344, f"b = f({minus(hs[i])}) = {bs[i]}", 18, TXT, hot, "center", 28)
    o += svg(0, 0, W, H, a)
    o += card(800, 150, 340, 180, "green", .2) + txt(808, 156, 324, t("輸出層：位元相加再減 0.5", "Output layer: add the bits, subtract 0.5"), 17, TXT, True, "center", 52) + formula(808, 212, 324, rf"q={bs[0]}+{bs[1]}+{bs[2]}-0.5={qq}", 22, 44) + chip(840, 270, 260, "green", t(f"q = {qq} > 0：判 A", f"q = {qq} > 0: call A"), True, 40, 20)
    o += txt(16, 430, 1128, t("橘色粗線是啟動的路徑；其他兩個偵測器輸出 0。〔程式計算〕", "The thick orange lines are the active path; the other two detectors output 0. (computed)"), 17, TXT, False, "center", 30)
    return o

# ------------------------------------------------ 6: OR merge on a number line
def page6(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 760, 480, "blue", t("輸出分數 q = k − 0.5 的數線", "The number line of the output score q = k − 0.5"))
    ax = Axes(60, 120, 660, 200, (-1, 3.5), (0, 1)); a = ""
    ax.rect(-1, 0.5 - 0.5, 0, 1, "#d97757", .0)
    a_x = lambda v: ax.X(v)
    ax.rect(-1, 0.0, 0.1, 0.9, "#d97757", .13); ax.rect(0.0, 3.5, 0.1, 0.9, "#6a9fd6", .13)
    ax.raw(f'<line x1="{a_x(-1)}" y1="{ax.Y(.45)}" x2="{a_x(3.5)}" y2="{ax.Y(.45)}" stroke="{GG}" stroke-width="3"/>')
    for kk in range(0, 4):
        qv = kk - 0.5; xx = a_x(kk - 0.5 + 0.5) if False else a_x(qv + 0.5)
    # axis in q: position q mapped to x = q + 0.5 so that k = 0..3 sit at 0.. wait -> use q directly
    ax2 = Axes(60, 120, 660, 200, (-1.5, 3.0), (0, 1)); ax = ax2; ax.o = []
    ax.rect(-1.5, 0.0, 0.1, 0.9, "#d97757", .13); ax.rect(0.0, 3.0, 0.1, 0.9, "#6a9fd6", .13)
    ax.raw(f'<line x1="{ax.X(-1.5)}" y1="{ax.Y(.45)}" x2="{ax.X(3.0)}" y2="{ax.Y(.45)}" stroke="{GG}" stroke-width="3"/>')
    ax.raw(f'<line x1="{ax.X(0)}" y1="{ax.Y(.05)}" x2="{ax.X(0)}" y2="{ax.Y(.95)}" stroke="{LINE}" stroke-width="3" stroke-dasharray="9 7"/>')
    for kk in range(0, 4):
        qv = kk - 0.5; col = CA if qv > 0 else CB
        ax.dot(qv, .45, col, 16); ax.text(qv, .45, f"k={kk}", "#f2ede4", "middle", 16, True, 0, -28); ax.text(qv, .45, minus(qv), "#f2ede4", "middle", 16, False, 0, 40)
    ax.text(0, .95, "q = 0", LINE, "middle", 17, True, 0, -6); ax.text(-0.75, .82, t("判 B", "call B"), CBt, "middle", 18, True); ax.text(1.5, .82, t("判 A", "call A"), CAt, "middle", 18, True)
    o += ax.html()
    o += card(20, 340, 720, 124, "yellow", .14) + txt(28, 346, 704, t("零點落在 k = 0 與 k = 1 之間：k = 0 判 B，k ≥ 1 判 A，這就是或合併。", "The zero lies between k = 0 and k = 1: k = 0 is B and k ≥ 1 is A, which is the OR merge."), 18, TXT, True, "center", 56) + txt(28, 408, 704, t("每個編碼最多讓一個偵測器啟動，實際只出現 k = 0 或 1。〔程式計算〕", "Each code fires at most one detector, so only k = 0 or 1 occur. (computed)"), 16, MUT, False, "center", 48)
    o += cardH(776, 0, 384, 480, "violet", t("合併只要一條規則", "One rule merges them"))
    o += card(792, 48, 352, 70, "violet", .22) + formula(792, 54, 352, r"q=b_{000}+b_{110}+b_{111}-0.5=k-0.5", 16.5, 58)
    o += txt(792, 126, 352, t("bₜ：偵測器的 0／1 輸出；k：輸出 1 的偵測器個數；q > 0 判 A。", "bₜ: the detector's 0/1 output; k: number of detectors outputting 1; q > 0 means A."), 16, TXT, False, "left", 76)
    for xx, lab, w_ in ((800, "k", 60), (866, "q = k − 0.5", 150), (1026, t("判定", "call"), 110)): o += txt(xx, 214, w_, lab, 16, MUT, True, "center", 24)
    for i in range(4):
        y0 = 242 + i * 42; v = i - 0.5; o += row(792, y0, 352, 36, "blue" if v > 0 else "orange", .2)
        o += txt(800, y0 + 5, 60, str(i), 17, TXT, False, "center", 26) + txt(866, y0 + 5, 150, minus(v), 17, TXT, True, "center", 26) + txt(1026, y0 + 5, 110, "A" if v > 0 else "B", 18, TXT, True, "center", 26)
    return o

# ------------------------------------------------ 7: all eight corners
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 1160, 470, "teal", t("八個角逐一檢查：只有 000、110、111 判 A", "All eight corners checked: only 000, 110, 111 are called A"))
    xs = [(14, 100, t("編碼 y", "code y")), (118, 130, "h₀₀₀"), (254, 130, "h₁₁₀"), (390, 130, "h₁₁₁"), (540, 190, t("位元 b", "bits b")), (740, 60, "k"), (806, 150, "q = k − 0.5"), (964, 90, t("判定", "call")), (1060, 90, t("真值", "truth"))]
    for xx, w_, lab in xs: o += txt(xx, 44, w_, lab, 16, MUT, True, "center", 24)
    for i, c in enumerate(K.CODES):
        y0 = 74 + i * 48; A = K.cls(c) == "A"; ghost = c == "101"
        o += row(10, y0, 1140, 42, "blue" if A else "grey", .2 if A else .08)
        o += txt(14, y0 + 8, 100, c, 18, TXT if not ghost else MUT, True, "center", 26)
        for j, tc in enumerate(K.TARGETS): o += txt(118 + j * 136, y0 + 8, 130, minus(K.h(tc, c)), 17, TXT, False, "center", 26)
        b = K.bits(c); o += txt(540, y0 + 8, 190, "(" + ", ".join(map(str, b)) + ")", 17, TXT, A, "center", 26) + txt(740, y0 + 8, 60, str(sum(b)), 17, TXT, False, "center", 26) + txt(806, y0 + 8, 150, minus(K.q(c)), 17, TXT, True, "center", 26) + txt(964, y0 + 8, 90, K.cls(c), 18, TXT, True, "center", 26)
        o += txt(1060, y0 + 8, 90, t("A" if c in K.TARGETS else "B", "A" if c in K.TARGETS else "B") + (t("（虛擬）", " (unused)") if ghost else ""), 17, MUT if ghost else TXT, False, "center", 26)
    return o

# ------------------------------------------------ 8: back in the input plane
def page8(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "blue", t("輸入平面：三條線切出 8 個編碼區", "The input plane: three lines cut 8 code regions"))
    ax = Axes(40, 52, 520, 392, (-3, 3), (-3, 3)); lo, hi = -3, 3
    def reg(xs, ys, col, op):
        s = " ".join(f"{ax.X(a_):.1f},{ax.Y(b_):.1f}" for a_, b_ in zip(xs, ys)); ax.raw(f'<polygon points="{s}" fill="{col}" opacity="{op}"/>')
    # regions via sampling polygons: draw full-plane cells with a fine grid of rects
    def clip(poly, g, keep):   # keep points with sign(g)==keep (g(x,y)>0 when keep=1)
        out = []
        for i in range(len(poly)):
            p, q_ = poly[i], poly[(i + 1) % len(poly)]; gp, gq = g(*p), g(*q_)
            ip = (gp > 0) == bool(keep); iq = (gq > 0) == bool(keep)
            if ip: out.append(p)
            if ip != iq: tt = gp / (gp - gq); out.append((p[0] + tt * (q_[0] - p[0]), p[1] + tt * (q_[1] - p[1])))
        return out
    gs = [lambda x, y: x, lambda x, y: y, lambda x, y: y - x - 1]
    for c in K.CODES:
        pl = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
        for g, ch in zip(gs, c): pl = clip(pl, g, int(ch))
        if len(pl) >= 3: reg([p[0] for p in pl], [p[1] for p in pl], "#6a9fd6" if K.cls(c) == "A" else "#d97757", .2)
    ax.line([(0, -3), (0, 3)], "#f0a35a", 4); ax.line([(-3, 0), (3, 0)], "#f0a35a", 4); ax.line([(-3, -2), (2, 3)], "#f0a35a", 4)
    for (px, py) in K.PA: ax.dot(px, py, CA, 14); ax.text(px, py, "A", "#f2ede4", "middle", 16, True, 0, 6)
    for (px, py) in K.PB: ax.rect(px - .17, px + .17, py - .17, py + .17, CB, 1); ax.text(px, py, "B", "#f2ede4", "middle", 16, True, 0, 6)
    for c, (px, py) in {"111": (0.9, 1.9), "110": (2.2, 1.2), "011": (-1.6, 1.8), "010": (-0.65, 0.45), "001": (-2.5, 0.35), "000": (-1.8, -1.2), "100": (1.5, -1.5)}.items(): ax.text(px, py, c, "#f2ede4", "middle", 16, True, 0, 0)
    ax.text(0.05, 2.9, "g₁ = 0", "#f0a35a", "start", 16, True, 6, 8); ax.text(2.95, -0.05, "g₂ = 0", "#f0a35a", "end", 16, True, 0, 20); ax.text(-2.95, -2.25, "g₃ = 0", "#f0a35a", "start", 16, True, 0, 0)
    o += ax.html()
    o += cardH(616, 0, 544, 480, "orange", t("三層網路把四個點都分對", "The three-layer network gets all four points right"))
    pr = [(p, "A") for p in K.PA] + [(p, "B") for p in K.PB]
    for xx, lab, w_ in ((628, t("點", "point"), 70), (704, t("編碼", "code"), 80), (790, t("輸出 q", "output q"), 100), (900, t("判定", "call"), 70), (976, t("真值", "truth"), 70)): o += txt(xx, 46, w_, lab, 16, MUT, True, "center", 24)
    for i, (p, tr_) in enumerate(pr):
        c = K.plane_code(*p); y0 = 74 + i * 38; ok = K.cls(c) == tr_; o += row(628, y0, 516, 34, "green" if ok else "red", .18)
        o += txt(628, y0 + 4, 70, f"{i+1}", 17, TXT, False, "center", 26) + txt(704, y0 + 4, 80, c, 17, TXT, True, "center", 26) + txt(790, y0 + 4, 100, minus(K.q(c)), 17, TXT, False, "center", 26) + txt(900, y0 + 4, 70, K.cls(c), 17, TXT, True, "center", 26) + txt(976, y0 + 4, 70, tr_ + " ✓", 17, TXT, False, "center", 26)
    o += txt(628, 232, 516, t("A 區是 000、110、111 三個編碼區的聯集（110 與 111 相鄰），B 區的 010 夾在其中。〔程式計算〕", "The A region is the union of the code regions 000, 110, 111 (110 and 111 are adjacent), with B's 010 squeezed in between. (computed)"), 16, TXT, False, "left", 76)
    o += card(628, 320, 516, 140, "red", .18) + txt(636, 326, 500, t("常見迷思：以為存在三層構造，網路就已學到權重、或這是最小的網路。", "Myth: that a three-layer construction exists means the weights are learned, or the network is minimal."), 17, TXT, True, "center", 56) + txt(636, 392, 500, t("構造只證明夠用；權重怎麼從帶標籤的點得到，是下一個知識點。", "The construction only shows it suffices; how weights come from labelled points is the next unit."), 16, TXT, False, "center", 56)
    return o

# ------------------------------------------------ 9: overlapping rectangles
def page9(lang):
    t = tr(lang); o = ""; s = 112; gx, gy = 140, 56
    o += cardH(0, 0, 620, 480, "violet", t("兩個重疊的長方形細分成 7 格", "Two overlapping rectangles refined into 7 cells"))
    a = ""
    def cellxy(c, r): return gx + c * s, gy + (2 - r) * s
    for c in range(3):
        for r in range(3):
            x, y = cellxy(c, r); inn = (c, r) in K.R1 | K.R2
            a += f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="{"#6a9fd6" if inn else "none"}" opacity="{.22 if inn else 1}" stroke="#6b645a" stroke-width="2" stroke-dasharray="6 5"/>'
    x, y = cellxy(0, 1); a += f'<rect x="{x}" y="{y}" width="{2*s}" height="{2*s}" fill="none" stroke="{OR}" stroke-width="6"/>'
    x, y = cellxy(1, 2); a += f'<rect x="{x}" y="{y}" width="{2*s}" height="{2*s}" fill="none" stroke="{VI}" stroke-width="6"/>'
    for i, (c, r) in enumerate(K.CELLS):
        x, y = cellxy(c, r); a += f'<text x="{x+s/2}" y="{y+s/2+8}" text-anchor="middle" font-size="22" font-weight="700" fill="#f2ede4">{i+1}</text>'
    o += svg(0, 0, W, H, a)
    o += txt(24, 412, 580, t("橘色與紫色框：兩個長方形；每個藍格是一個編碼區。", "Orange and purple frames: the two rectangles; each blue cell is one code region."), 16, TXT, False, "center", 26)
    o += cardH(636, 0, 524, 480, "orange", t("偵測器的個數看格子，不看長方形", "Detectors count cells, not rectangles"))
    o += row(652, 52, 492, 90, "grey", .12) + txt(660, 58, 476, t("兩個長方形各用 4 條線，共 8 條線，把平面切成格子。", "Each rectangle uses 4 lines: 8 lines in all cut the plane into cells."), 17, TXT, False, "center", 76)
    o += row(652, 154, 492, 100, "blue", .2) + txt(660, 160, 476, t(f"A 佔其中 {len(K.CELLS)} 格：需要 {len(K.CELLS)} 個偵測器，不是 2 個。〔程式計算〕", f"A occupies {len(K.CELLS)} cells: {len(K.CELLS)} detectors are needed, not 2. (computed)"), 19, TXT, True, "center", 84)
    o += card(652, 266, 492, 190, "yellow", .14) + txt(660, 274, 476, t("原長方形的個數不等於細分後的編碼數；做法不變：每個編碼區配一個偵測器，再或合併。", "The number of original rectangles is not the number of codes after refining; the recipe is unchanged: one detector per code region, then an OR merge."), 18, TXT, False, "center", 160)
    return o

# ------------------------------------------------ 10: exercises
def page10(lang):
    t = tr(lang); o = ""; (sv, bias), a1, a2, a3 = K.practice()
    qs = [("blue", t("題 1", "Q1"), t("寫出偵測 011 的 h₀₁₁，並算 y = 011 與 y = 111 的分數。", "Write h₀₁₁ for detecting 011 and compute its score at y = 011 and y = 111."), [r"h_{011}=-y_1+y_2+y_3-1.5", r"y=011:\ 0+1+1-1.5=0.5", r"y=111:\ -1+1+1-1.5=-0.5"], t("011 輸出 1，111 輸出 0。", "011 outputs 1 and 111 outputs 0."), t("核對：011 有兩個 1，權重 (−1, +1, +1)，偏置 −(2 − 0.5) = −1.5。", "Check: 011 has two 1s, weights (−1, +1, +1),<br>bias −(2 − 0.5) = −1.5.")),
          ("red", t("題 2", "Q2"), t("y = 010 的分數是多少？先用「每錯一位少 1」猜，再代入核對。", "What is the score at y = 010? Guess with 'every wrong bit costs 1', then substitute."), [r"010\ \text{vs}\ 011:\ m=1\Rightarrow 0.5-1=-0.5", r"\text{substitute}:\ 0+1+0-1.5=-0.5"], t("分數 −0.5，輸出 0。", "Score −0.5, output 0."), t("核對：兩種算法一致。〔程式計算〕", "Check: the two ways agree. (computed)"))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 50, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 146, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 190, 538, 150, "grey", .1)
        for j, tx in enumerate(tex): o += formula(x + 16, 198 + j * (150 // len(tex)) + (6 if len(tex) == 2 else 0), 538, tx, 22 if len(tex) == 3 else 21, 48)
        o += chip(x + 16, 356, 538, role, ans, True, 42, 19)
        o += txt(x + 16, 410, 538, chk, 16, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 11)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp04_" + fid, fn, W, H, out)
