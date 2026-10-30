import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP01"; N = 8
X = (1, 0, 1); Z = sum(X); assert Z == 2
TH = (1, 2, 3); S = [Z - t for t in TH]; O = [1 if s >= 0 else 0 for s in S]
assert S == [1, 0, -1] and O == [1, 1, 0]
AND_S = [x1 + x2 - 2 for x2 in (0, 1) for x1 in (0, 1)]; OR_S = [x1 + x2 - 1 for x2 in (0, 1) for x1 in (0, 1)]
assert AND_S == [-2, -1, -1, 0] and OR_S == [-1, 0, 0, 1]
NOT_S = [-1 * x + 0.5 for x in (0, 1)]; assert NOT_S == [0.5, -0.5]

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)
    sg = lambda s: ("+" if s > 0 else "") + str(s).replace("-", "−")

    # ============ 1: biological neuron ============
    g = new(1, T("KP01 神經元加總訊號，過門檻才放電", "KP01 A neuron adds signals, fires past a threshold"))
    g.beat = 1
    ys = (180, 330, 480)
    for i, y in enumerate(ys):
        g.line([(130, y), (400, 330)], BLUE, sw=5, z=3)
        g.rect(40, y - 24, 90, 48, fill=BLUEL, stroke=BLUE, sw=3, radius=12, z=4); g.text(40, y - 24, 90, 48, f"$x_{i + 1}$", size=24, color=BLUE, bold=True, align="center", vcenter=True, z=8)
    g.text(40, 540, 380, 70, T("樹突:每一路帶一個訊號", "dendrites: each carries one signal"), size=20, color=BLUE, bold=True, align="center", z=8)
    g.beat = 2
    g.ellipse(520, 330, 130, 110, fill=ORGL, stroke=ORG, sw=4, z=3); g.text(390, 280, 260, 100, r"$\Sigma$", size=60, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.text(390, 450, 260, 70, T("細胞體:把訊號加在一起", "soma: adds the signals"), size=20, color=ORG, bold=True, align="center", z=8)
    g.beat = 3
    g.line([(650, 330), (930, 330)], GRN, sw=7, arrow="end", head=20, z=3)
    g.text(660, 270, 260, 40, T("軸突:送出脈衝", "axon: sends the pulse"), size=20, color=GRN, bold=True, align="center", z=8)
    g.ellipse(960, 330, 28, fill=GRNL, stroke=GRN, sw=4, z=3); g.text(900, 370, 150, 60, T("突觸", "synapse"), size=20, color=GRN, bold=True, align="center", z=8)
    g.ellipse(1130, 330, 80, 70, fill=CARD, stroke=GRAY, sw=4, z=3); g.text(1050, 300, 160, 60, T("下一顆神經元", "next neuron"), size=18, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    g.line([(990, 330), (1050, 330)], GRAY, sw=4, arrow="end", head=14, z=3)
    star(g, 560, 220, 20, GOLD, z=6)
    banner(g, T("累積 → 過門檻 → 放電;不是形狀,是機制", "accumulate → pass the threshold → fire: the mechanism, not the shape"))
    pages.append((g, T("生物神經元", "A biological neuron"), {
        0: T("人工神經元模仿的是生物神經元的機制。先認識四個部件。", "An artificial neuron copies the mechanism of a biological one. First, four parts."),
        1: T("樹突像天線，每一路接收一個訊號。", "Dendrites act like antennas; each receives one signal."),
        2: T("細胞體把收到的訊號加在一起。細胞核在它裡面。", "The soma adds the received signals together, with the nucleus inside."),
        3: T("只有總量跨過門檻，才沿軸突放電，經突觸傳給下一顆神經元；沒跨過就完全沉默。", "Only when the total crosses a threshold does it fire along the axon and, through the synapse, to the next neuron; otherwise it stays silent.")}))

    # ============ 2: switch, not knob ============
    g = new(2, T("它是開關，不是音量鈕", "It is a switch, not a volume knob"))
    U = 150; X0 = 330; TH2 = 2
    g.beat = 1
    for i, z in enumerate((1, 2, 3)):
        y = 150 + i * 130
        g.text(60, y, 250, 70, T(f"總和 $z={z}$", f"total $z={z}$"), size=26, color=INK, bold=True, vcenter=True, z=8)
        g.rect(X0, y + 10, z * U, 50, fill=BLUEL, stroke=BLUE, sw=3, radius=10, z=3)
    g.line([(X0 + TH2 * U, 120), (X0 + TH2 * U, 540)], RED, sw=5, dash=True, z=5)
    g.text(X0 + TH2 * U - 100, 548, 200, 40, T("門檻 $\\theta=2$", "threshold $\\theta=2$"), size=22, color=RED, bold=True, align="center", z=8)
    g.beat = 2
    for i, z in enumerate((1, 2, 3)):
        y = 150 + i * 130; fire = z >= TH2
        g.rect(900, y + 6, 310, 58, fill=GRNL if fire else REDL, stroke=GRN if fire else RED, sw=3, radius=14, z=3)
        g.text(900, y + 6, 310, 58, T("輸出 1 放電", "output 1: fires") if fire else T("輸出 0 沉默", "output 0: silent"), size=22, color=GRN if fire else RED, bold=True, align="center", vcenter=True, z=8)
        g.line([(X0 + z * U + 14, y + 35), (888, y + 35)], GRAY, sw=4, arrow="end", head=14, z=3)
    g.beat = 3
    banner(g, T("$z=2$ 與 $z=3$ 都輸出 1:輸出只問有沒有跨過門檻", "$z=2$ and $z=3$ both give 1: the output only asks whether the bar was crossed"))
    pages.append((g, T("開關不是音量鈕", "Switch, not knob"), {
        0: T("同一個門檻，三種總和。橫條長度就是總和 z。", "One threshold, three totals. The bar length is the total z."),
        1: T("總和 1 沒到門檻，沉默；總和 2 剛好到，放電；總和 3 超過，也放電。", "Total 1 misses the bar and stays silent; total 2 just reaches it and fires; total 3 exceeds it and also fires."),
        2: T("z 等於 2 和 3 輸出一樣。輸出只回答有沒有跨過門檻，不是輸入越強輸出越大。", "z = 2 and z = 3 give the same output. The output only says whether the threshold was crossed; stronger input does not give a bigger output."),
        3: T("所以它是開關。", "So it is a switch.")}))

    # ============ 3: formula ============
    g = new(3, T("把生物機制寫成一條式子", "Write the mechanism as one formula"))
    rows = [(BLUE, BLUEL, T("多路訊號", "many signals"), r"$x_1,\dots,x_d$", T("輸入", "inputs")), (ORG, ORGL, T("每一路的重要性", "importance of each"), r"$w_1,\dots,w_d$", T("權重", "weights")),
            (PUR, PURL, T("門檻", "threshold"), r"$w_0=-\theta$", T("偏置", "bias")), (GRN, GRNL, T("放電或沉默", "fire or stay silent"), r"$f(s)$", T("活化函數", "activation"))]
    for i, (col, light, bio, sym, nm) in enumerate(rows):
        g.beat = 1 if i < 2 else 2
        y = 120 + i * 100
        a = g.mark(); g.rect(60, y, 440, 80, fill=CARD, stroke=col, sw=3, radius=14, z=3); g.text(60, y, 440, 80, bio, size=24, color=col, bold=True, align="center", vcenter=True, z=8)
        b2 = g.mark(); g.rect(640, y, 580, 80, fill=light, stroke=col, sw=3, radius=14, z=3)
        g.text(650, y, 270, 80, sym, size=28, color=INK, align="center", vcenter=True, z=8); g.text(920, y, 290, 80, nm, size=24, color=col, bold=True, align="center", vcenter=True, z=8)
        g.link(a, b2, 4, 12, col, 4)
    g.beat = 3
    card(g, 190, 556, 900, 76, GOLD); g.text(200, 556, 880, 76, r"$o=f\!\left(\sum_{j=1}^{d}w_jx_j+w_0\right),\quad f(s)=1\ (s\ge0),\ 0\ (s<0)$", size=28, color=INK, align="center", vcenter=True, z=8)
    pages.append((g, T("加權和、偏置、活化", "Weighted sum, bias, activation"), {
        0: T("把生物機制逐項對到數學元件。多路訊號寫成輸入，每一路的重要性寫成權重。", "Map each biological piece to a mathematical one. Signals become inputs and the importance of each becomes a weight."),
        1: T("多路訊號寫成輸入 x，每一路的重要性寫成權重 w。", "Signals become inputs x; the importance of each becomes a weight w."),
        2: T("門檻搬進偏置，w0 等於負的 theta；放電或沉默寫成活化函數 f。", "The threshold moves into the bias, w0 equals minus theta; fire or stay silent becomes the activation f."),
        3: T("合起來就是 1943 年 McCulloch 與 Pitts 的神經元：加權和加偏置，再過階梯函數。總和大於等於 theta 與 s 大於等於 0 是同一個條件。", "Together this is the 1943 McCulloch–Pitts neuron: weighted sum plus bias, then a step. Sum at least theta and s at least 0 are the same condition.")}))

    # ============ 4: hand computation ============
    g = new(4, T("手算:同一組輸入，只換門檻", "By hand: same inputs, different thresholds"))
    g.beat = 1
    for i, v in enumerate(X):
        x = 80 + i * 130
        g.ellipse(x + 40, 190, 40, fill=BLUEL if v else CARD, stroke=BLUE, sw=3, z=3); g.text(x, 150, 80, 80, str(v), size=30, color=BLUE, bold=True, align="center", vcenter=True, z=8)
        g.text(x, 238, 80, 30, f"$x_{i + 1}$", size=20, color=MUTED, align="center", z=8)
    g.line([(470, 190), (560, 190)], GRAY, sw=5, arrow="end", head=16, z=3)
    g.ellipse(640, 190, 70, fill=ORGL, stroke=ORG, sw=4, z=3); g.text(570, 150, 140, 80, r"$z=2$", size=30, color=ORG, bold=True, align="center", vcenter=True, z=8)
    g.text(740, 150, 480, 80, r"$z=1+0+1=2$" + T("(權重全為 1)", " (all weights 1)"), size=24, color=INK, bold=True, vcenter=True, z=8)
    g.beat = 2
    hd = [r"$\theta$", r"$w_0=-\theta$", r"$s=z+w_0$", r"$o=f(s)$"]
    cx = [60, 330, 640, 960]; cw = [250, 290, 300, 250]
    for k, h in enumerate(hd):
        g.text(cx[k], 290, cw[k], 44, h, size=24, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    for i, t in enumerate(TH):
        y = 345 + i * 80; hl = (t == 2)
        g.rect(50, y, 1180, 68, fill=("#fef3c7" if hl else CARD), stroke=(GOLD if hl else GRAY), sw=3, radius=14, z=3)
        vals = [str(t), sg(-t) if False else f"${-t}$", rf"$2{'-' if t > 0 else '+'}{t}={S[i]}$", None]
        g.text(cx[0], y, cw[0], 68, str(t), size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(cx[1], y, cw[1], 68, f"${-t}$", size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(cx[2], y, cw[2], 68, f"$2-{t}={S[i]}$", size=28, color=INK, bold=True, align="center", vcenter=True, z=8)
        g.text(cx[3], y, cw[3], 68, str(O[i]), size=32, color=GRN if O[i] else RED, bold=True, align="center", vcenter=True, z=8)
    g.beat = 3
    g.rect(240, 584, 800, 44, fill="#fef3c7", stroke=GOLD, sw=3, radius=22, z=3); g.text(240, 584, 800, 44, T("$\\theta=2$:$s=0$ 剛好達標，仍然放電", "$\\theta=2$: $s=0$ just reaches the bar, so it still fires"), size=22, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("總和不變，只有門檻在變:輸出 1、1、0", "the total never changes, only the threshold: outputs 1, 1, 0"))
    pages.append((g, T("手算三個門檻", "Three thresholds by hand"), {
        0: T("主例子：三路輸入 1、0、1，權重全為 1。總和 z 等於 2，輸入是 0 的那一路不貢獻。", "Main example: three inputs 1, 0, 1 with all weights 1. The total z is 2; the input 0 contributes nothing."),
        1: T("門檻 1：s 等於 1，輸出 1。門檻 2：s 等於 0，輸出 1。門檻 3：s 等於負 1，輸出 0。", "Threshold 1: s is 1, output 1. Threshold 2: s is 0, output 1. Threshold 3: s is minus 1, output 0."),
        2: T("門檻 2 時 s 等於 0 剛好達標，所以仍然放電，這就是階梯函數用大於等於 0 的原因。", "At threshold 2, s is 0 and just reaches the bar, so the neuron still fires; this is why the step uses at least 0."),
        3: T("總和沒變，只有門檻在變。", "The total never changed; only the threshold did.")}))

    # ============ 5: step function ============
    g = new(5, T("階梯函數:$s\\ge0$ 就放電", "The step function: fire when $s\\ge0$"))
    ox, oy, sc = 640, 430, 150
    g.beat = 1
    g.line([(130, oy), (1160, oy)], INK, sw=4, arrow="end", head=16, z=3); g.line([(ox, 490), (ox, 150)], INK, sw=4, arrow="end", head=16, z=3)
    g.text(1100, oy + 8, 80, 36, "$s$", size=24, color=INK, z=8); g.text(ox + 14, 140, 80, 36, "$o$", size=24, color=INK, z=8)
    g.text(ox - 60, oy + 8, 50, 30, "0", size=18, color=MUTED, align="right", z=8)
    g.line([(190, oy), (ox, oy)], BLUE, sw=7, z=4); g.line([(ox, 240), (1100, 240)], BLUE, sw=7, z=4)
    g.ellipse(ox, 240, 12, fill=BLUE, stroke=BLUE, sw=3, z=6); g.ellipse(ox, oy, 12, fill=CARD, stroke=BLUE, sw=4, z=6)
    g.text(ox + 14, 244 - 6, 160, 30, "$o=1$", size=20, color=BLUE, bold=True, z=8) if False else None
    g.beat = 2
    for s_, o_ in zip(S, O):
        px = ox + s_ * sc; py = 240 if o_ else oy
        col = GRN if o_ else RED
        g.ellipse(px, py, 18, fill=col, stroke="#ffffff", sw=3, z=7)
        g.text(px - 90 if s_ != 0 else px - 200, py - 62 if o_ else py + 24, 180 if s_ != 0 else 190, 40, f"$s={s_}$, $o={o_}$", size=22, color=col, bold=True, align="center" if s_ != 0 else "right", z=8)
        g.line([(px, oy + 4), (px, oy + 20)], GRAY, sw=3, z=3) if False else None
    g.text(ox - 150, 160, 300, 40, T("放電", "fire"), size=22, color=GRN, bold=True, align="center", z=8) if False else None
    g.beat = 3
    g.rect(150, 530, 980, 84, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(170, 530, 940, 84, T("用 $s\\ge0$ 還是 $s>0$ 只是約定不同;同一份計算要一致", "$s\\ge0$ versus $s>0$ is only a convention; keep one in a whole computation"), size=24, color=INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("階梯函數:左邊一律 0，從 $s=0$ 起一律 1", "the step: 0 on the left, 1 from $s=0$ onward"))
    pages.append((g, T("階梯函數", "The step function"), {
        0: T("活化函數是階梯：s 小於 0 輸出 0，s 大於等於 0 輸出 1。實心點在 s 等於 0 的上方，空心點在下方。", "The activation is a step: output 0 when s is below 0, output 1 when s is at least 0. The solid dot sits above s equals 0, the open dot below."),
        1: T("把上一頁三個門檻的 s 放上來：s 等於 1 與 0 在放電那一段，s 等於負 1 在沉默那一段。", "Put the three s values from the last page on it: s equal 1 and 0 are on the firing level, s equal minus 1 on the silent level."),
        2: T("用大於等於還是大於，只是約定不同，同一份計算要一致。本單元統一用大於等於。", "At least versus greater than is only a convention; use one throughout. This unit uses at least."),
        3: T("左邊一律 0，從 0 起一律 1。", "Zero on the left, one from zero on.")}))

    # ============ 6: AND / OR ============
    g = new(6, T("只改偏置，同一顆神經元變 AND 或 OR", "Only the bias changes: AND or OR"))
    pts = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for pn, (ox_, nm, th, ss, col) in enumerate(((60, "AND", 2, AND_S, BLUE), (680, "OR", 1, OR_S, ORG))):
        g.beat = pn + 1
        u = lambda v: ox_ + 120 + v * 200
        w_ = lambda v: 440 - v * 200
        g.rect(ox_, 110, 540, 410, fill=CARD, stroke=col, sw=3, radius=16, z=1)
        g.text(ox_, 112, 540, 44, f"{nm}:" + T(f"$w_0=-{th}$", f"$w_0=-{th}$"), size=26, color=col, bold=True, align="center", vcenter=True, z=8)
        # decision line x1 + x2 = th  (within box)
        a_ = (th - 1.35, 1.35) if th == 2 else (th - 1.2, 1.2); b_ = (a_[1], a_[0])
        g.line([(u(a_[0]), w_(a_[1])), (u(b_[0]), w_(b_[1]))], RED, sw=5, dash=True, z=5)
        for (x1, x2), s_ in zip(pts, ss):
            o_ = 1 if s_ >= 0 else 0
            g.ellipse(u(x1), w_(x2), 24, fill=(GRN if o_ else CARD), stroke=(GRN if o_ else RED), sw=4, z=6)
            g.text(u(x1) + 30, w_(x2) - 14, 110, 30, f"$s={s_}$", size=18, color=MUTED, bold=True, z=8) if x1 == 0 else g.text(u(x1) - 30 - 100 if x1 == 1 and False else u(x1) + 30, w_(x2) - 14, 110, 30, f"$s={s_}$", size=18, color=MUTED, bold=True, z=8)
    g.beat = 3
    g.text(60, 534, 1160, 40, T("實心:輸出 1　空心:輸出 0　虛線:$s=0$ 的決策線", "solid: output 1 · open: output 0 · dashed: the decision line $s=0$"), size=20, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("同一顆神經元，$w_0=-2$ 是 AND，$w_0=-1$ 是 OR", "one neuron: $w_0=-2$ is AND, $w_0=-1$ is OR"))
    pages.append((g, T("AND 與 OR", "AND and OR"), {
        0: T("兩個輸入、權重全為 1，偏置取負 2：只有兩個輸入都是 1，s 才達到 0，這是 AND。四個角的 s 是負 2、負 1、負 1、0。", "Two inputs, all weights 1, bias minus 2: s reaches 0 only when both inputs are 1; that is AND. The four corners have s equal minus 2, minus 1, minus 1, 0."),
        1: T("偏置取負 1：只要有一個輸入是 1，s 就達到 0，這是 OR。四個角的 s 是負 1、0、0、1。", "Bias minus 1: s reaches 0 as soon as one input is 1; that is OR. The four corners have s equal minus 1, 0, 0, 1."),
        2: T("虛線就是 s 等於 0 的決策線。同一顆神經元，只差偏置，就是不同的邏輯閘。", "The dashed line is the decision line s equals 0. The same neuron, differing only in bias, is a different gate."),
        3: T("但權重和偏置是人事先算好的。", "But a person worked out the weights and bias beforehand.")}))

    # ============ 7: NOT ============
    g = new(7, T("NOT 閘需要負權重", "A NOT gate needs a negative weight"))
    g.beat = 1
    for i, x in enumerate((0, 1)):
        y = 140 + i * 140; s_ = NOT_S[i]; o_ = 1 if s_ >= 0 else 0
        g.beat = 1 + i
        a = g.mark(); g.rect(60, y, 220, 90, fill=BLUEL, stroke=BLUE, sw=3, radius=14, z=3); g.text(60, y, 220, 90, f"$x={x}$", size=30, color=BLUE, bold=True, align="center", vcenter=True, z=8)
        b2 = g.mark(); g.rect(420, y, 400, 90, fill=ORGL, stroke=ORG, sw=3, radius=14, z=3); g.text(420, y, 400, 90, rf"$s=-1\cdot{x}+0.5={s_}$", size=28, color=ORG, bold=True, align="center", vcenter=True, z=8)
        c = g.mark(); g.rect(960, y, 260, 90, fill=GRNL if o_ else REDL, stroke=GRN if o_ else RED, sw=3, radius=14, z=3); g.text(960, y, 260, 90, f"$o={o_}$", size=30, color=GRN if o_ else RED, bold=True, align="center", vcenter=True, z=8)
        g.link(a, b2, 4, 12, GRAY, 4); g.link(b2, c, 4, 12, GRAY, 4)
    g.beat = 3
    g.rect(120, 430, 1040, 170, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(150, 442, 980, 60, T("輸入由 0 變 1，輸出要由 1 變 0", "When the input goes 0 → 1 the output must go 1 → 0"), size=26, color=RED, bold=True, align="center", z=8)
    g.text(150, 506, 980, 90, T("所以 $s$ 必須變小:權重得是負的。權重全為正的神經元，輸入變大只會讓 $s$ 變大，做不出 NOT。", "so $s$ must shrink: the weight must be negative. With all weights positive, a bigger input can only grow $s$, so NOT is impossible."), size=26, color=INK, bold=True, align="center", z=8)
    banner(g, T("NOT:$w=-1$，$w_0=0.5$", "NOT: $w=-1$, $w_0=0.5$"))
    pages.append((g, T("NOT 閘", "The NOT gate"), {
        0: T("NOT 只有一個輸入。取權重負 1、偏置 0.5。", "NOT has one input. Take weight minus 1 and bias 0.5."),
        1: T("輸入 0：s 等於 0.5，輸出 1。", "Input 0: s is 0.5, output 1."),
        2: T("輸入 1：s 等於負 0.5，輸出 0。輸入變大，s 變小，所以權重必須是負的；權重全為正的神經元做不出 NOT。", "Input 1: s is minus 0.5, output 0. A bigger input lowers s, so the weight must be negative; a neuron with all-positive weights cannot make NOT."),
        3: T("權重的正負，決定這一路是促進還是抑制。", "The sign of a weight decides whether that input promotes or inhibits.")}))

    # ============ 8: wrap-up ============
    g = new(8, T("這顆神經元缺的是學習", "What this neuron lacks is learning"))
    g.beat = 1
    card(g, 50, 120, 1180, 100, BLUE); g.text(70, 120, 1140, 100, r"$o=f\!\left(\sum_{j=1}^{d}w_jx_j+w_0\right)$", size=36, color=INK, align="center", vcenter=True, z=8)
    pts3 = [T("加總訊號，過門檻才放電", "add the signals, fire only past a threshold"), T("只改閾值或權重:AND、OR、NOT", "change only the threshold or weights: AND, OR, NOT"), T("McCulloch 與 Pitts:連成網路能算任何邏輯函數", "McCulloch and Pitts: a network of them computes any logic function")]
    for i, s_ in enumerate(pts3):
        g.beat = 2
        y = 250 + i * 80
        tick(g, 70, y + 14, 34, GRN, z=6); g.text(130, y, 900, 60, s_, size=24, color=INK, bold=True, vcenter=True, z=8)
    g.beat = 3
    g.rect(50, 500, 1000, 120, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 540 if zh else 520, 960, 90, T("權重與閾值都要人事先算好。能不能只給資料與正確答案，讓它自己把權重改對？", "A person must work out the weights and threshold beforehand. Can we give only data and right answers and let it fix its own weights?"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1100, 420, 0.55, "confused")
    banner(g, T("下一個單元:Perceptron 讓權重自己學", "next unit: the Perceptron learns its own weights"))
    pages.append((g, T("本單元重點與問題", "Summary and question"), {
        0: T("整個單元都在說這一條式子。", "The whole unit is about this one formula."),
        1: T("重點三件事：加總訊號、過門檻才放電；只改閾值或權重，就是 AND、OR、NOT；McCulloch 與 Pitts 證明連成網路可以算出任何邏輯函數。", "Three points: add signals and fire only past a threshold; changing only the threshold or weights gives AND, OR, NOT; McCulloch and Pitts showed networks of them compute any logic function."),
        2: T("但權重要人事先算好。能不能只給資料與正確答案，讓它自己改對？下一個單元的 Perceptron 就是答案。", "But a person must work out the weights. Can we give only data and right answers and let it correct itself? The next unit's Perceptron is the answer."),
        3: T("神經元已經有計算力，缺的是學習。", "The neuron already has computing power; what it lacks is learning.")}))
    return pages
