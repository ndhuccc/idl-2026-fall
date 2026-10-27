"""KP06 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
import kp06_nums as N
W, H = 1160, 480
def box(x, y, w, h, c1, c2, ang=135, shadow=True, r=14):
    return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-radius:{r}px;background:linear-gradient({ang}deg,{c1},{c2});' + ('box-shadow:0 10px 18px 0 rgba(0,0,0,0.45);' if shadow else '') + '"></div>'
def cardH(x, y, w, h, role, title, size=17): return card(x, y, w, h, role) + head(x, y, w, role, title, 36, size)
def sg(v, d=3):
    s = f"{v:.{d}f}"; s = s.rstrip("0").rstrip(".") if "." in s else s
    s = "0" if s in ("-0", "") else s; return s.replace("-", "−")
def ss(v, d=4): return ("+" if v > 0 else "") + sg(v, d)
def circ(cx, cy, r, role, s):
    return f'<div style="position:absolute;left:{cx-r}px;top:{cy-r}px;width:{2*r}px;height:{2*r}px;border-radius:50%;background:{rgb(role)};color:{INK};font-size:19px;font-weight:700;text-align:center;line-height:{2*r}px">{s}</div>'
POS = {"x1": (60, 110), "x2": (60, 290), "h1": (260, 70), "h2": (260, 330), "o": (460, 200)}
def net(x0, y0, labels=None, back=None, wts=True):
    """2-2-1 network of the lecture example at (x0, y0). labels: node -> text under/over the node. back: dict of delta texts to draw red back arrows."""
    P = {k: (x0 + a, y0 + b) for k, (a, b) in POS.items()}; Wt = N.W; s = ""; labs = ""
    edges = [("x1", "h1", Wt["h1"][1], .36), ("x2", "h1", Wt["h1"][2], .70), ("x1", "h2", Wt["h2"][1], .28), ("x2", "h2", Wt["h2"][2], .36), ("h1", "o", Wt["out"][1], .40), ("h2", "o", Wt["out"][2], .40)]
    for a, b, w, f in edges:
        (xa, ya), (xb, yb) = P[a], P[b]; col = "#7fb0e6" if w >= 0 else "#e08a7a"
        s += f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}" stroke="{col}" stroke-width="{2 + abs(w) * 3:.1f}" opacity="0.85"/>'
        lx, ly = xa + (xb - xa) * f, ya + (yb - ya) * f; labs += f'<div style="position:absolute;left:{lx-30:.0f}px;top:{ly-14:.0f}px;width:60px;height:26px;background:#1c1712;color:{TXT};font-size:17px;font-weight:700;text-align:center;line-height:26px;border-radius:6px">{("−" if w < 0 else "") + f"{abs(w):.2f}"}</div>'
    if back:
        for h_, (dx, dy) in (("h1", (0, -10)), ("h2", (0, 10))):
            (xa, ya), (xb, yb) = P["o"], P[h_]; ux, uy = (xb - xa), (yb - ya); L = math.hypot(ux, uy); ux, uy = ux / L, uy / L
            sx, sy, ex, ey = xa - ux * 34 + dx * 0.0, ya - uy * 34 + 12 * (1 if h_ == "h2" else -1), xb + ux * -34, yb + 12 * (1 if h_ == "h2" else -1)
            s += f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" stroke="#e0605a" stroke-width="4" stroke-dasharray="10 6"/>'
            ang = math.atan2(ey - sy, ex - sx); hx, hy = ex, ey; s += f'<polygon points="{hx:.0f},{hy:.0f} {hx - 16*math.cos(ang - .4):.0f},{hy - 16*math.sin(ang - .4):.0f} {hx - 16*math.cos(ang + .4):.0f},{hy - 16*math.sin(ang + .4):.0f}" fill="#e0605a"/>'
    o = svg(0, 0, W, H, s) + labs
    for k, role, lab in (("x1", "grey", "x₁"), ("x2", "grey", "x₂"), ("h1", "blue", "h₁"), ("h2", "blue", "h₂"), ("o", "orange", "o")): o += circ(P[k][0], P[k][1], 30, role, lab)
    for k, b in (("h1", N.W["h1"][0]), ("h2", N.W["h2"][0]), ("o", N.W["out"][0])):
        cx, cy = P[k]; o += txt(cx - 70, cy + (-62 if k != "h2" else 34), 140, "b = " + sg(b, 2), 16, "#e0c85f", True, "center", 24)
    for k, (dxl, dyl, text, col) in (labels or {}).items():
        cx, cy = P[k]; o += txt(cx + dxl, cy + dyl, 150, text, 17, col, True, "center", 26)
    return o

# ---------------------------------------------------- 1: the step has no slope
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 376, 420, "red", t("階梯函數：斜率處處為 0", "Step function: slope 0 everywhere"))
    ax = Axes(36, 66, 316, 270, (-3, 3), (-0.3, 1.4)); ax.frame([-2, 0, 2], [0, 1], yfmt=lambda v: f"{v:g}")
    ax.line([(-3, 0), (0, 0)], OR, 5).line([(0, 1), (3, 1)], OR, 5).line([(0, 0), (0, 1)], MUT, 2, "7 6")
    ax.raw(f'<circle cx="{ax.X(0):.1f}" cy="{ax.Y(0):.1f}" r="7" fill="#1c1712" stroke="#e08a7a" stroke-width="3"/><circle cx="{ax.X(0):.1f}" cy="{ax.Y(1):.1f}" r="7" fill="#e0605a"/>')
    ax.text(-1.5, 0, t("斜率 0", "slope 0"), "#e08a7a", "middle", 17, True, dy=-14).text(1.5, 1, t("斜率 0", "slope 0"), "#e08a7a", "middle", 17, True, dy=-14).text(0.2, 0.5, t("跳躍處沒有斜率", "jump: no slope"), TXT, "start", 16, True, dx=8)
    o += ax.html() + txt(10, 368, 356, t("拿不到斜率，梯度下降無從下手。", "No slope to read: gradient descent has nothing to work with."), 16, TXT, False, "center", 48)
    o += cardH(392, 0, 376, 420, "teal", t("sigmoid：每一點都有斜率", "Sigmoid: a slope at every point"))
    ax = Axes(428, 66, 316, 270, (-3, 3), (-0.3, 1.4)); ax.frame([-2, 0, 2], [0, 0.5, 1], yfmt=lambda v: f"{v:g}")
    xs = np.linspace(-3, 3, 50); ax.line([(float(x), N.f(float(x))) for x in xs], TE, 5).line([(-1.8, 0.5 - 0.25 * 1.8), (1.8, 0.5 + 0.25 * 1.8)], YE, 3, "8 6")
    ax.dot(0, 0.5, YE, 8).text(0.3, 1.2, t("切線斜率 0.25", "tangent slope 0.25"), "#e0c85f", "start", 17, True, dx=-20)
    o += ax.html() + txt(402, 368, 356, t("在 x = 0：f′ = f(1 − f) = 0.5 × 0.5 = 0.25。", "At x = 0: f′ = f(1 − f) = 0.5 × 0.5 = 0.25."), 16, TXT, False, "center", 48)
    o += card(0, 430, 768, 50, "yellow", .14) + txt(8, 438, 752, t("要讓網路自己學權重，就得把階梯換成處處有斜率的函數。", "To learn weights, the step must become a function with a slope everywhere."), 17, TXT, True, "center", 34)
    o += cardH(784, 0, 376, 480, "orange", t("名詞", "Terms"))
    o += named(796, 46, 352, t("代價函數：把預測和目標差多少加總成一個數，越小越好。本講義用 E = ½(t − o)²。", "Cost function: add up how far the prediction is from the target into one number, the smaller the better. Here E = ½(t − o)²."), "term:" + t("代價函數", "Cost function"), 16, TXT, 120)
    o += named(796, 190, 352, t("梯度下降：看代價對每個權重的斜率，往讓代價變小的方向把權重調一點。", "Gradient descent: look at the slope of the cost with respect to each weight and nudge the weight in the direction that lowers the cost."), "term:" + t("梯度下降", "Gradient descent"), 16, TXT, 120)
    o += card(796, 350, 352, 114, "yellow", .14) + txt(804, 358, 336, t("KP05 的權重是規則直接給的；這一章要讓網路自己學。", "The weights of KP05 were given by rules; this chapter lets the network learn them."), 16, TXT, False, "center", 96)
    return o

# ---------------------------------------------------- 2: sigmoid and a
def page2(lang):
    t = tr(lang); o = ""; As = (0.7, 2, 8); cols = {0.7: OR, 2: TE, 8: VI}
    o += cardH(0, 0, 400, 480, "blue", t("sigmoid 與斜率參數 a", "Sigmoid and the slope parameter a"))
    o += formula(8, 44, 384, r"f(x)=\frac{1}{1+e^{-ax}}", 24, 62) + formula(8, 112, 384, r"f'(x)=a\,f(x)\,(1-f(x))", 22, 46)
    o += named(14, 170, 372, t("sigmoid：把任何數壓進 0 到 1 之間；a 越大，曲線越像階梯。", "Sigmoid: squeezes any number into 0 to 1; larger a, closer to a step."), "term:sigmoid", 16, TXT, 76)
    o += card(14, 266, 372, 198, "red", .14) + txt(22, 272, 356, t("常見迷思：越接近階梯越好。其實 a 越大，斜率的高峰越窄越高，兩側的斜率幾乎是 0，反而越難用梯度訓練。", "Myth: the closer to a step the better. In fact the larger a is, the narrower and higher the slope peak and the flatter both sides, so training by gradient gets harder."), 16, TXT, True, "left", 180)
    o += cardH(416, 0, 366, 480, "teal", t("a = 0.7、2、8 與虛線階梯", "a = 0.7, 2, 8 and the dashed step"))
    ax = Axes(448, 60, 312, 330, (-4, 4), (-0.1, 1.15)); ax.frame([-4, -2, 0, 2, 4], [0, 0.5, 1], yfmt=lambda v: f"{v:g}")
    ax.line([(-4, 0), (0, 0)], MUT, 2, "7 6").line([(0, 1), (4, 1)], MUT, 2, "7 6")
    xs = np.linspace(-4, 4, 90)
    for a in As: ax.line([(float(x), N.f(float(x), a)) for x in xs], cols[a], 4)
    o += ax.html() + legend([(cols[a], f"a = {a:g}") for a in As], 640, 240, 100)
    o += cardH(798, 0, 362, 480, "violet", t("斜率 f′：高峰越來越窄", "Slope f′: the peak gets narrower"))
    ax = Axes(830, 60, 312, 330, (-4, 4), (0, 2.2)); ax.frame([-4, -2, 0, 2, 4], [0, 1, 2], yfmt=lambda v: f"{v:g}")
    for a in As: ax.line([(float(x), N.fp(float(x), a)) for x in xs], cols[a], 4)
    o += ax.html() + txt(806, 436, 346, t("高峰 a/4：0.175、0.5、2。", "Peaks a/4: 0.175, 0.5, 2."), 16, TXT, True, "center", 28)
    return o

# ---------------------------------------------------- 3: forward pass
def page3(lang):
    t = tr(lang); g = N.grads(N.W); o = ""
    o += cardH(0, 0, 540, 480, "blue", t("2–2–1 網路：輸入 x = (1, 0)，目標 t = 1", "A 2–2–1 network: input x = (1, 0), target t = 1"))
    o += net(10, 56, {"h1": (90, -40, "y₁ = %.3f" % g["y1"], "#7fb0e6"), "h2": (90, 14, "y₂ = %.3f" % g["y2"], "#7fb0e6"), "o": (-62, 40, "o = %.3f" % g["o"], "#e8a56a"), "x1": (0, -62, "x₁ = 1", MUT), "x2": (0, 36, "x₂ = 0", MUT)})
    o += txt(14, 436, 512, t("b 是偏置；線上的數字是權重。", "b is the bias; the numbers on the lines are weights."), 16, MUT, False, "center", 24)
    o += cardH(556, 0, 604, 480, "green", t("前向傳遞：一層一層往前算並存起來", "Forward pass: compute layer by layer and store the values"))
    o += named(566, 44, 584, t("前向傳遞：從輸入一層一層往前算；每個單元先算加權和 v，再套 sigmoid 得 y = f(v)，並把中間值存下來。", "Forward pass: compute from the input layer by layer; each unit forms the weighted sum v, applies the sigmoid to get y = f(v), and stores the values."), "term:" + t("前向傳遞", "Forward pass"), 16, TXT, 76)
    rows = [(r"v_1=-0.10+0.20\cdot1+0.40\cdot0=0.10", r"y_1=f(0.10)=%.3f" % g["y1"], "blue"), (r"v_2=0.10-0.30\cdot1+0.20\cdot0=-0.20", r"y_2=f(-0.20)=%.3f" % g["y2"], "blue"), (r"v_o=-0.20+0.50\cdot%.3f-0.40\cdot%.3f=%.3f" % (g["y1"], g["y2"], g["vo"]), r"o=f(%.3f)=%.3f" % (g["vo"], g["o"]), "orange")]
    for i, (tex, tex2, role) in enumerate(rows):
        y0 = 124 + i * 74; o += card(566, y0, 584, 68, role, .16) + formula(566, y0 + 2, 584, tex, 18, 30) + formula(566, y0 + 36, 584, tex2, 18, 28)
    o += card(566, 352, 584, 56, "yellow", .16) + formula(566, 354, 584, r"e=1-%.3f=%.3f,\ \ E=\frac{1}{2}e^2=%.3f" % (g["o"], g["e"], N.E(N.W)), 22, 52)
    o += txt(566, 422, 584, t("輸出 0.471 離目標 1 還差 0.529。", "The output 0.471 is still 0.529 short of the target 1."), 17, TXT, True, "center", 28)
    return o

# ---------------------------------------------------- 4: chain rule
def page4(lang):
    t = tr(lang); g = N.grads(N.W); o = ""
    o += cardH(0, 0, 760, 480, "orange", t("把 w(o1) 的梯度拆成三段斜率相乘", "Split the gradient of w(o1) into three slopes"))
    xs = [30, 220, 410, 600]; nm = [("w", "grey"), ("v", "blue"), ("o", "orange"), ("E", "red")]
    for x, (lab, role) in zip(xs, nm): o += circ(x + 55, 120, 34, role, lab)
    ar = ""
    for i in range(3): ar += f'<line x1="{xs[i]+95}" y1="120" x2="{xs[i+1]+11}" y2="120" stroke="#b7b0a5" stroke-width="3"/><polygon points="{xs[i+1]+11},120 {xs[i+1]-3},113 {xs[i+1]-3},127" fill="#b7b0a5"/>'
    o += svg(0, 0, W, H, ar)
    sl = [(r"\frac{\partial v_o}{\partial w_{o1}}=y_1", "= %.3f" % g["y1"], "grey"), (r"\frac{\partial o}{\partial v_o}=o(1-o)", "= %.3f" % g["fpo"], "blue"), (r"\frac{\partial E}{\partial o}=-(t-o)", "= %s" % sg(-g["e"]), "red")]
    for i, (tex, val, role) in enumerate(sl):
        x = 22 + i * 240; o += card(x, 180, 226, 148, role, .16) + formula(x, 190, 226, tex, 20, 64) + txt(x, 258, 226, val, 24, TXT, True, "center", 36) + txt(x, 296, 226, [t("本權重乘的輸入", "the input of this weight"), t("sigmoid 的斜率", "slope of the sigmoid"), t("平方的斜率再乘 −1", "slope of the square, times −1")][i], 16, MUT, False, "center", 24)
    o += card(22, 346, 716, 120, "yellow", .16) + formula(22, 356, 716, r"\frac{\partial E}{\partial w_{o1}}=(-0.529)\times0.249\times0.525=%.3f" % (-g["e"] * g["fpo"] * g["y1"]), 24, 56) + txt(30, 414, 700, t("梯度是負的：把 w(o1) 調大，E 會變小。〔程式計算〕", "The gradient is negative: raising w(o1) lowers E. (computed)"), 17, TXT, True, "center", 40)
    o += cardH(776, 0, 384, 480, "violet", t("名詞", "Terms"))
    o += named(788, 46, 360, t("連鎖律：A 影響 B、B 影響 C 時，C 對 A 的斜率等於 C 對 B 的斜率乘上 B 對 A 的斜率。", "Chain rule: when A affects B and B affects C, the slope of C with respect to A is the slope of C with respect to B times the slope of B with respect to A."), "term:" + t("連鎖律", "Chain rule"), 16, TXT, 140)
    o += formula(780, 200, 372, r"\frac{\partial E}{\partial w}=\frac{\partial E}{\partial o}\cdot\frac{\partial o}{\partial v}\cdot\frac{\partial v}{\partial w}", 22, 70)
    o += txt(788, 296, 360, t("梯度（代價對一個權重的斜率）是負的，表示把這個權重調大會讓代價變小。", "A negative gradient (the slope of the cost with respect to one weight) means raising that weight lowers the cost."), 16, TXT, False, "left", 96)
    return o

# ---------------------------------------------------- 5: delta and output update
def page5(lang):
    t = tr(lang); g = N.grads(N.W); o = ""
    o += cardH(0, 0, 580, 480, "violet", t("誤差訊號 δ：前兩段合起來只算一次", "Error signal δ: the first two slopes combined, computed once"))
    o += card(14, 52, 552, 76, "violet", .22) + formula(14, 60, 552, r"\delta_o=e\,f'(v_o)=0.529\times0.249=%.4f" % g["do"], 23, 60)
    o += named(18, 140, 544, t("誤差訊號 δ：δ ≡ −∂E/∂v，也就是「誤差乘上斜率」；它會沿權重往回傳。", "Error signal δ: δ ≡ −∂E/∂v, the “error times slope”; it is passed back along the weights."), "term:" + t("誤差訊號", "Error signal"), 16, TXT, 76)
    o += chip(18, 232, 544, "blue", t("梯度：∂E/∂w = −δ × 輸入（例：−0.069 = −0.1319 × 0.525）", "Gradient: ∂E/∂w = −δ × input (e.g. −0.069 = −0.1319 × 0.525)"), False, 40, 16)
    o += chip(18, 280, 544, "green", t("下降方向：Δw = η δ × 輸入（加上去）", "Descent: Δw = η δ × input (add it)"), False, 40, 16)
    o += named(18, 334, 544, t("學習率 η：每次調整權重的步子大小，本例 η = 0.3。", "Learning rate η: the size of each adjustment step; here η = 0.3."), "term:" + t("學習率", "Learning rate"), 16, TXT, 30)
    o += card(14, 376, 552, 88, "yellow", .14) + txt(22, 382, 536, t("來源把 δ 同時寫成 ∂E/∂v 與「誤差乘斜率」，差一個負號；本講義統一 δ = −∂E/∂v。", "The source writes δ both as ∂E/∂v and as “error times slope”, which differ by a sign; this lecture fixes δ = −∂E/∂v."), 16, TXT, False, "left", 76)
    o += cardH(596, 0, 564, 480, "green", t("三個輸出權重的更新量 η δ × 輸入", "The three output-weight updates η δ × input"))
    ins = [(1, t("偏置", "bias")), (g["y1"], "y₁ = %.3f" % g["y1"]), (g["y2"], "y₂ = %.3f" % g["y2"])]; mx = 0.045
    for i, (iv, lab) in enumerate(ins):
        v = N.ETA * g["do"] * iv; x0 = 640 + i * 168; hgt = v / mx * 280; top = 400 - hgt
        o += box(x0, top, 120, hgt, "#6ed09a", "#2c6140", 180, shadow=False, r=10) + txt(x0 - 14, top - 34, 148, "+%.4f" % v, 21, TXT, True, "center", 30) + txt(x0 - 14, 408, 148, lab, 16, MUT, True, "center", 24)
    o += txt(606, 442, 544, t("三根都是正的：目標 1 比輸出 0.471 大，三個權重都該調大。", "All positive: the target 1 is above the output 0.471, so all weights should grow."), 16, TXT, False, "center", 24)
    return o

# ---------------------------------------------------- 6: backward
def page6(lang):
    t = tr(lang); g = N.grads(N.W); o = ""
    o += cardH(0, 0, 540, 480, "red", t("誤差訊號沿權重往回走", "The error signal travels back along the weights"))
    o += net(10, 56, {"o": (-62, 40, "δₒ = %.4f" % g["do"], "#e8a56a"), "h1": (96, -44, "δ₁ = %.4f" % g["d1"], "#e0c85f"), "h2": (96, 20, "δ₂ = %s" % sg(g["d2"], 4), "#e0c85f")}, back=True)
    o += txt(14, 436, 512, t("紅色虛線：誤差從輸出往回走的路徑。", "Red dashed lines: the path of the error from the output back."), 16, MUT, False, "center", 24)
    o += cardH(556, 0, 604, 480, "orange", t("隱藏層看不到目標：把 δ 沿著權重往回分", "Hidden units never see the target: pass δ back along the weights"))
    o += named(566, 44, 584, t("反向傳播：前向算完所有輸出後，從輸出層往回走；隱藏單元的 δ 是下一層每個 δ 乘連過去的權重後加總，再乘上自己的斜率。", "Backpropagation: after the forward pass, walk back from the output layer; a hidden unit's δ is the sum of the next layer's δ times the connecting weights, times its own slope."), "term:" + t("反向傳播", "Backpropagation"), 16, TXT, 100)
    o += card(566, 156, 584, 70, "orange", .2) + formula(566, 164, 584, r"\delta_j=f'(v_j)\sum_k\delta_k\,w_{kj}", 24, 54)
    o += card(566, 238, 584, 62, "blue", .16) + formula(566, 244, 584, r"\delta_1=0.2494\times0.1319\times0.50=%.4f" % g["d1"], 21, 50)
    o += card(566, 312, 584, 62, "red", .16) + formula(566, 318, 584, r"\delta_2=0.2475\times0.1319\times(-0.40)=%.4f" % g["d2"], 21, 50)
    o += txt(566, 386, 584, t("w(o2) 是負的，所以 δ₂ < 0；k 只有輸出單元一個。", "w(o2) is negative, so δ₂ < 0; k runs over the single output unit."), 16, TXT, False, "left", 48)
    o += card(566, 432, 584, 38, "yellow", .14) + txt(574, 436, 568, t("索引 j：隱藏單元；k：下一層；w(kj)：j 連到 k 的權重。", "Index j: hidden unit; k: next layer; w(kj): the weight from j to k."), 16, TXT, False, "center", 28)
    return o

# ---------------------------------------------------- 7: nine weights, before/after
def page7(lang):
    t = tr(lang); g = N.grads(N.W); W2, _ = N.update(N.W); I, D = N.inputs(g), N.deltas(g); o = ""
    o += cardH(0, 0, 860, 480, "green", t("九個權重各調一步（η = 0.3）", "Nine weights, one step each (η = 0.3)"))
    cols = [(10, 120, t("權重", "weight")), (134, 100, t("舊值", "old")), (240, 70, t("輸入", "input")), (316, 100, t("所屬 δ", "its δ")), (422, 110, t("梯度", "gradient")), (540, 120, t("更新量", "update")), (666, 100, t("新值", "new"))]
    for x, w_, lab in cols: o += txt(x, 44, w_, lab, 16, MUT, True, "center", 24)
    names = [("out", 0, t("輸出 偏置", "out bias")), ("out", 1, t("輸出 w₁", "out w₁")), ("out", 2, t("輸出 w₂", "out w₂")), ("h1", 0, t("隱藏 1 偏置", "hid 1 bias")), ("h1", 1, t("隱藏 1 x₁", "hid 1 x₁")), ("h1", 2, t("隱藏 1 x₂", "hid 1 x₂")), ("h2", 0, t("隱藏 2 偏置", "hid 2 bias")), ("h2", 1, t("隱藏 2 x₁", "hid 2 x₁")), ("h2", 2, t("隱藏 2 x₂", "hid 2 x₂"))]
    for i, (k, j, nm) in enumerate(names):
        y0 = 74 + i * 40; w0 = N.W[k][j]; inp = I[k][j]; dl = D[k]; gr = -dl * inp; up = N.ETA * dl * inp; hot = abs(up) > 0.01
        o += card(10, y0, 840, 34, "green" if hot else "grey", .2 if hot else .08)
        vals = [(10, 120, nm, True), (134, 100, sg(w0, 4), False), (240, 70, sg(inp, 3), False), (316, 100, sg(dl, 4), False), (422, 110, ss(gr, 4) if gr else "0", False), (540, 120, ss(up, 4) if abs(up) > 5e-6 else "0", True), (666, 100, sg(W2[k][j], 4), False)]
        for x, w_, s_, b_ in vals: o += txt(x, y0 + 3, w_, s_, 16, TXT, b_, "center", 26)
    o += txt(10, 440, 840, t("x₂ = 0，所以兩個 x₂ 的權重這次不動。〔程式計算〕", "x₂ = 0, so the two x₂ weights do not move this time. (computed)"), 16, MUT, False, "left", 24)
    o += cardH(876, 0, 284, 480, "yellow", t("更新前後", "Before and after"))
    o += txt(884, 52, 268, t("輸出 o", "output o"), 17, MUT, True, "center", 26) + chip(890, 84, 116, "grey", "0.4706", False, 40, 19) + txt(1004, 90, 36, "→", 20, TXT, True, "center", 28) + chip(1040, 84, 104, "green", "%.4f" % N.fwd(W2)["o"], True, 40, 19)
    o += txt(884, 160, 268, t("代價 E", "cost E"), 17, MUT, True, "center", 26) + chip(890, 192, 116, "grey", "0.1401", False, 40, 19) + txt(1004, 198, 36, "→", 20, TXT, True, "center", 28) + chip(1040, 192, 104, "green", "%.4f" % N.E(W2), True, 40, 19)
    o += txt(884, 264, 268, t("目標 t = 1：輸出往目標走了一小步，代價從 0.140 降到 0.132。一步只走一小步，要重複很多步。", "Target t = 1: the output moved a small step toward it and the cost fell from 0.140 to 0.132. One step is small; many are needed."), 16, TXT, False, "left", 144)
    return o

# ---------------------------------------------------- 8: numeric check
def page8(lang):
    t = tr(lang); g = N.grads(N.W); I, D = N.inputs(g), N.deltas(g); o = ""
    names = [("out", 0, "o<br>b"), ("out", 1, "o<br>w₁"), ("out", 2, "o<br>w₂"), ("h1", 0, "h₁<br>b"), ("h1", 1, "h₁<br>x₁"), ("h1", 2, "h₁<br>x₂"), ("h2", 0, "h₂<br>b"), ("h2", 1, "h₂<br>x₁"), ("h2", 2, "h₂<br>x₂")]
    o += cardH(0, 0, 760, 480, "teal", t("九個權重：−δ × 輸入（長條）與差分斜率（圓點）", "Nine weights: −δ × input (bars) and finite-difference slopes (dots)"))
    ax = Axes(70, 70, 670, 300, (0, 9), (-0.16, 0.06)); ax.raw(f'<line x1="{ax.x}" y1="{ax.y}" x2="{ax.x}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/><line x1="{ax.x}" y1="{ax.Y(0):.1f}" x2="{ax.x+ax.w}" y2="{ax.Y(0):.1f}" stroke="#b7b0a5" stroke-width="2"/>')
    for v in (-0.15, -0.1, -0.05, 0): ax.raw(f'<text x="{ax.x-8}" y="{ax.Y(v)+5:.1f}" text-anchor="end" font-size="16" fill="#b7b0a5">{sg(v, 2)}</text>')
    b = ""; gw = ax.w / 9
    for i, (k, j, nm) in enumerate(names):
        an = -D[k] * I[k][j]; nu = N.numeric(N.W, k, j); bx = ax.x + i * gw + 12; y0 = ax.Y(0); yy = ax.Y(an)
        top, hh = (min(y0, yy), max(abs(yy - y0), 2))
        b += box(bx, top, gw - 24, hh, "#5fb8a8", "#2f7d70", 180, shadow=False, r=6)
        ax.dot((i + 0.5), nu, YE, 7) if False else ax.raw(f'<circle cx="{ax.x + (i+0.5)*gw:.1f}" cy="{ax.Y(nu):.1f}" r="8" fill="#e0c85f" stroke="#1c1712" stroke-width="2"/>')
        b += txt(ax.x + i * gw + 8, 378, gw - 16, nm, 16, MUT, True, "center", 48)
    o += ax.html() + b
    o += legend([("#5fb8a8", t("−δ × 輸入", "−δ × input")), ("#e0c85f", t("差分斜率", "difference slope"))], 330, 84, 240)
    o += txt(14, 432, 732, t("兩者重合，反向傳播的算式正確。", "The two coincide: the backpropagation formulas are right."), 17, TXT, True, "center", 28)
    o += cardH(776, 0, 384, 480, "orange", t("中心差分", "Central difference"))
    o += named(788, 46, 360, t("中心差分：把某個權重加減一點點 ε，看 E 變多少，用兩邊的差除以 2ε 當斜率。", "Central difference: nudge one weight by ±ε, see how much E changes and divide the two-sided difference by 2ε to get a slope."), "term:" + t("中心差分", "Central difference"), 16, TXT, 100)
    o += card(788, 160, 360, 82, "orange", .2) + formula(788, 168, 360, r"\frac{E(w+\varepsilon)-E(w-\varepsilon)}{2\varepsilon},\ \varepsilon=10^{-6}", 20, 64)
    mx = max(abs(-D[k] * I[k][j] - N.numeric(N.W, k, j)) for k, j, _ in names)
    o += card(788, 256, 360, 100, "green", .16) + txt(796, 262, 344, t("九個權重的最大差距約 %s〔程式計算〕" % sci(mx).replace("×10", "×10"), "Largest gap over the nine weights about %s (computed)" % sci(mx)), 17, TXT, True, "center", 84)
    o += txt(788, 370, 360, t("這是寫反向傳播程式時最常用的除錯方法。", "This is the most common debugging method when writing a backpropagation program."), 16, MUT, False, "left", 72)
    return o

# ---------------------------------------------------- 9: saturation
def page9(lang):
    t = tr(lang); o = ""; vals = [(a, N.fp(3, a)) for a in (0.7, 2, 8)]; cs = [("#e8a56a", "#a8452b"), ("#5fb8a8", "#2f7d70"), ("#a98bd0", "#5a4a8a")]
    o += cardH(0, 0, 640, 480, "violet", t("x = 3 處的斜率 f′(3)（對數座標）", "The slope f′(3) at x = 3 (log scale)"))
    ax = Axes(100, 60, 500, 330, (0, 3), (-10.6, 0)); ax.raw(f'<line x1="{ax.x}" y1="{ax.y}" x2="{ax.x}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/><line x1="{ax.x}" y1="{ax.y+ax.h}" x2="{ax.x+ax.w}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/>')
    for e_ in (0, -2, -4, -6, -8, -10): py = ax.Y(e_); ax.raw(f'<line x1="{ax.x}" y1="{py:.1f}" x2="{ax.x+ax.w}" y2="{py:.1f}" stroke="#6b645a" stroke-width="1" opacity="0.45"/><text x="{ax.x-8}" y="{py+5:.1f}" text-anchor="end" font-size="16" fill="#b7b0a5">10{_sup(e_)}</text>')
    b = ""
    for i, ((a, v), (c1, c2)) in enumerate(zip(vals, cs)):
        bx = ax.x + i * 166 + 30; top = ax.Y(math.log10(v)); b += box(bx, top, 100, ax.y + ax.h - top, c1, c2, 180, shadow=False, r=8) + txt(bx - 20, top - 32, 140, sci(v), 20, TXT, True, "center", 30) + txt(bx - 10, ax.y + ax.h + 8, 120, f"a = {a:g}", 18, TXT, True, "center", 26)
    o += ax.html() + b
    o += cardH(656, 0, 504, 480, "red", t("斜率飽和", "Slope saturation"))
    o += named(668, 46, 480, t("斜率飽和：a 太大時，離中心不遠 f′ 就幾乎是 0；δ 乘上幾乎是 0 的斜率，權重幾乎不動。", "Slope saturation: when a is too large, f′ is almost 0 not far from the centre; δ multiplied by an almost-zero slope barely moves the weights."), "term:" + t("斜率飽和", "Slope saturation"), 16, TXT, 100)
    o += formula(662, 160, 492, r"f'(3)=a\,f(1-f)", 22, 44)
    for i, (a, v) in enumerate(vals): o += chip(668, 214 + i * 50, 480, ["orange", "teal", "violet"][i], f"a = {a:g}：f′(3) = {sci(v)}", False, 40, 18)
    o += card(668, 372, 480, 92, "yellow", .16) + txt(676, 378, 464, t("a = 8 比 a = 0.7 小了約 8 個數量級；所以 a 取中等大小。", "a = 8 is about 8 orders of magnitude below a = 0.7; so a is chosen at a middle size."), 17, TXT, True, "center", 72)
    return o

# ---------------------------------------------------- 10: exercises
def page10(lang):
    t = tr(lang); g0 = N.grads(N.W, 0); I = N.inputs(g0)["out"]; ups = [N.ETA * g0["do"] * i for i in I]; o = ""
    qs = [("blue", t("題 1", "Q1"), t("沿用本例（a = 1、η = 0.3），只把目標從 1 改成 0，前向傳遞不變。誤差 e 與輸出層的 δ 是多少？", "Same example (a = 1, η = 0.3) but with the target changed from 1 to 0; the forward pass is unchanged. What are e and the output δ?"),
           [r"e=0-0.471=%.3f" % g0["e"], r"\delta_o=%.3f\times0.249=%.4f" % (g0["e"], g0["do"])], t("e = −0.471；δ = −0.1173。", "e = −0.471; δ = −0.1173."), t("核對：輸出 0.471 要降向 0，δ 該是負的。", "Check: the output 0.471 must fall toward 0, so δ should be negative.")),
          ("red", t("題 2", "Q2"), t("三個輸出權重各往哪個方向調、調多少（η = 0.3）？", "Which way and by how much does each of the three output weights move (η = 0.3)?"),
           [r"\Delta w=0.3\times(%.4f)\times\text{input}" % g0["do"], r"\text{bias}:%.4f\quad w_1:%.4f\quad w_2:%.4f" % tuple(ups)], t("偏置 −0.0352、w₁ −0.0185、w₂ −0.0158，三個都往下調。", "bias −0.0352, w₁ −0.0185, w₂ −0.0158; all three move down."), t("核對：和目標為 1 時符號剛好相反。", "Check: the signs are exactly the opposite of the target-1 case."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 50, 538, q, 18, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 192, 538, 156, "grey", .1) + formula(x + 16, 200, 538, tex[0], 22 if i == 0 else 20, 62) + formula(x + 16, 274, 538, tex[1], 22 if i == 0 else 17, 62)
        o += chip(x + 16, 362, 538, role, ans, True, 42, 18)
        o += txt(x + 16, 416, 538, chk, 17, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 11)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp06_" + fid, fn, W, H, out)
