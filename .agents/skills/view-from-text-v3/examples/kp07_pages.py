"""KP07 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
import kp07_nums as N, kp07_data as D
W, H = 1160, 480
def box(x, y, w, h, c1, c2, ang=135, shadow=True, r=14):
    return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-radius:{r}px;background:linear-gradient({ang}deg,{c1},{c2});' + ('box-shadow:0 10px 18px 0 rgba(0,0,0,0.45);' if shadow else '') + '"></div>'
def cardH(x, y, w, h, role, title, size=17): return card(x, y, w, h, role) + head(x, y, w, role, title, 36, size)
def sg(v, d=3):
    s = f"{v:.{d}f}"; s = s.rstrip("0").rstrip(".") if "." in s else s
    s = "0" if s in ("-0", "") else s; return s.replace("-", "−")
def circ(cx, cy, r, role, s):
    return f'<div style="position:absolute;left:{cx-r}px;top:{cy-r}px;width:{2*r}px;height:{2*r}px;border-radius:50%;background:{rgb(role)};color:{INK};font-size:19px;font-weight:700;text-align:center;line-height:{2*r}px">{s}</div>'
def net(x0, y0, Wt, outs=None):
    """2-2-1 network at (x0, y0) in a 520x400 area. Wt: 3x3 weights (bias, in1, in2)."""
    pos = {"x1": (60, 120), "x2": (60, 280), "h1": (260, 80), "h2": (260, 308), "o": (460, 195)}; P = {k: (x0 + a, y0 + b) for k, (a, b) in pos.items()}
    s = ""; labs = ""
    edges = [("x1", "h1", Wt[0][1], .36), ("x2", "h1", Wt[0][2], .70), ("x1", "h2", Wt[1][1], .28), ("x2", "h2", Wt[1][2], .36), ("h1", "o", Wt[2][1], .35), ("h2", "o", Wt[2][2], .35)]
    for a, b, w, f in edges:
        (xa, ya), (xb, yb) = P[a], P[b]; col = "#7fb0e6" if w >= 0 else "#e08a7a"
        s += f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}" stroke="{col}" stroke-width="{2 + min(abs(w), 12) * 0.25:.1f}" opacity="0.85"/>'
        lx, ly = xa + (xb - xa) * f, ya + (yb - ya) * f; labs += f'<div style="position:absolute;left:{lx-38:.0f}px;top:{ly-15:.0f}px;width:76px;height:26px;background:#1c1712;color:{TXT};font-size:17px;font-weight:700;text-align:center;line-height:26px;border-radius:6px">{sg(w)}</div>'
    o = svg(0, 0, W, H, s) + labs
    for k, role, lab in (("x1", "grey", "x₁"), ("x2", "grey", "x₂"), ("h1", "blue", "h₁"), ("h2", "blue", "h₂"), ("o", "orange", "o")): o += circ(P[k][0], P[k][1], 30, role, lab)
    for k, b in (("h1", Wt[0][0]), ("h2", Wt[1][0]), ("o", Wt[2][0])):
        cx, cy = P[k]; dy = -62 if k != "h2" else 36
        o += txt(cx - 70, cy + dy, 140, "b = " + sg(b), 16, "#e0c85f", True, "center", 24)
    return o

# ---------------------------------------------------- 1: the loop
def page1(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 720, 480, "blue", t("訓練重複四個步驟", "Training repeats four steps"))
    st = [("grey", "①", t("起始值", "Start values"), t("每個權重給一個小的隨機值（只做一次）", "give each weight a small random value (once only)")),
          ("blue", "②", t("前向", "Forward"), t("算出每個單元的輸出並存起來", "compute and store every unit's output")),
          ("orange", "③", t("反向", "Backward"), t("算每一筆的誤差訊號 δ", "compute the error signal δ of every sample")),
          ("green", "④", t("更新", "Update"), t("把各筆的 δ × 輸入加總，更新權重", "sum the δ × input of all samples and update the weights"))]
    xs = [24, 24, 254, 484]; ys = [64, 232, 232, 232]
    for (role, no, name, d), x, y in zip(st, xs, ys):
        o += card(x, y, 214, 150 if y > 100 else 120, role, .2) + chip(x + 10, y + 10, 194, role, f"{no} {name}", True, 34, 19) + txt(x + 8, y + 52, 198, d, 16, TXT, False, "center", 96 if y > 100 else 66)
    ar = f'<polygon points="238,300 252,307 238,314" fill="#b7b0a5"/><polygon points="468,300 482,307 468,314" fill="#b7b0a5"/><line x1="238" y1="307" x2="252" y2="307" stroke="#b7b0a5" stroke-width="3"/><line x1="468" y1="307" x2="482" y2="307" stroke="#b7b0a5" stroke-width="3"/>'
    ar += '<line x1="591" y1="384" x2="591" y2="410" stroke="#e0c85f" stroke-width="3"/><line x1="591" y1="410" x2="131" y2="410" stroke="#e0c85f" stroke-width="3"/><line x1="131" y1="410" x2="131" y2="392" stroke="#e0c85f" stroke-width="3"/><polygon points="124,396 131,384 138,396" fill="#e0c85f"/>'
    ar += '<line x1="131" y1="184" x2="131" y2="222" stroke="#b7b0a5" stroke-width="3"/><polygon points="124,220 131,232 138,220" fill="#b7b0a5"/>'
    o += svg(0, 0, W, H, ar)
    o += txt(240, 418, 340, t("重複 ②③④，直到停止", "repeat ②③④ until it stops"), 17, "#e0c85f", True, "center", 28)
    o += txt(260, 440, 440, t("例：XOR 的 N = 4 筆資料", "Example: XOR has N = 4 samples"), 16, MUT, False, "center", 24)
    o += cardH(736, 0, 424, 480, "orange", t("名詞", "Terms"))
    o += named(748, 48, 400, t("epoch：把全部 N 筆訓練資料呈現完一次。本例 N = 4，所以看完四個點算一個 epoch。", "Epoch: present all N training samples once. Here N = 4, so one pass over the four points is one epoch."), "term:" + t("epoch", "Epoch"), 16, TXT, 96)
    o += named(748, 160, 400, t("批次訓練：一個 epoch 裡每筆都先算出 δ，加總後只更新一次；除以 N 只是改變 η 的尺度。", "Batch training: compute δ for every sample in the epoch, add them up and update once; dividing by N only rescales η."), "term:" + t("批次訓練", "Batch training"), 16, TXT, 96)
    o += card(748, 282, 400, 110, "orange", .2) + formula(748, 300, 400, r"\Delta w=\frac{\eta}{N}\sum_{i=1}^{N}\delta(i)\,y^{r-1}(i)", 22, 70)
    o += txt(748, 404, 400, t("y 是第 i 筆在上一層的輸出，偏置的輸入固定是 1。", "y is the output of sample i in the layer below; the input of a bias is always 1."), 16, MUT, False, "left", 48)
    return o

# ---------------------------------------------------- 2: epoch 1 forward
def page2(lang):
    t = tr(lang); rows, dW, J0, J1 = D.epoch1(); o = ""
    o += cardH(0, 0, 540, 480, "blue", t("起始權重（來源互動例）：2–2–1 網路", "Start weights (source demo): a 2–2–1 network"))
    o += net(10, 50, N.W0)
    o += txt(14, 446, 512, t("b 是偏置；線上的數字是權重。三個單元都是 sigmoid（a = 1）。", "b: bias; line numbers: weights; all units sigmoid (a = 1)."), 16, MUT, False, "center", 48)
    o += cardH(556, 0, 604, 480, "green", t("四個點先各自前向（目標 0、1、1、0）", "The four points go forward first (targets 0, 1, 1, 0)"))
    cols = [(566, 84, t("輸入", "input")), (654, 90, "h₁"), (748, 90, "h₂"), (842, 90, "o"), (936, 80, t("目標", "target")), (1020, 130, "½(t − o)²")]
    for xx, ww, lab in cols: o += txt(xx, 48, ww, lab, 16, MUT, True, "center", 24)
    Js = []
    for i, (x, y, h, ou, do, dh) in enumerate(rows):
        e = 0.5 * (y - ou) ** 2; Js.append(e); y0 = 80 + i * 56; o += card(566, y0, 584, 48, "green" if abs(y - ou) < 0.5 else "grey", .14)
        o += txt(566, y0 + 9, 84, f"({int(x[0])}, {int(x[1])})", 17, TXT, True, "center", 28) + txt(654, y0 + 9, 90, f"{h[0]:.4f}", 17, TXT, False, "center", 28) + txt(748, y0 + 9, 90, f"{h[1]:.4f}", 17, TXT, False, "center", 28)
        o += txt(842, y0 + 9, 90, f"{ou:.4f}", 17, TXT, True, "center", 28) + txt(936, y0 + 9, 80, str(int(y)), 17, TXT, False, "center", 28) + txt(1020, y0 + 9, 130, f"{e:.5f}", 17, TXT, False, "center", 28)
    o += card(566, 312, 584, 78, "yellow", .16) + formula(566, 322, 584, (r"J=\frac{1}{4}(" + "+".join(f"{x:.5f}" for x in Js) + r")=%.5f") % J0, 19, 58)
    o += txt(566, 400, 584, t("四個點都已在 0.5 的正確一側：(0,0)、(1,1) 的輸出小於 0.5，(0,1)、(1,0) 大於 0.5。", "All four points are already on the correct side of 0.5: the outputs of (0,0) and (1,1) are below 0.5, those of (0,1) and (1,0) above."), 16, TXT, False, "left", 72)
    return o

# ---------------------------------------------------- 3: epoch 1 backward and update
def page3(lang):
    t = tr(lang); rows, dW, J0, J1 = D.epoch1(); o = ""
    o += cardH(0, 0, 560, 480, "orange", t("反向：各筆輸出層的 δ（輸出單元）", "Backward: the δ of the output unit for each sample"))
    o += formula(8, 44, 544, r"\delta_o=(t-o)\,o\,(1-o)", 22, 42)
    for i, (x, y, h, ou, do, dh) in enumerate(rows):
        y0 = 100 + i * 54; v = f"{do:+.4f}".replace("-", "−") if abs(do) >= 5e-5 else f"{do:+.5f}".replace("-", "−"); o += card(14, y0, 532, 46, "red" if do < -0.01 else "grey", .18)
        o += txt(20, y0 + 8, 84, f"({int(x[0])}, {int(x[1])})", 16, TXT, True, "left", 30) + txt(104, y0 + 8, 64, f"t = {int(y)}", 16, TXT, False, "left", 30) + txt(180, y0 + 8, 160, f"o = {ou:.4f}", 17, TXT, False, "center", 30) + chip(360, y0 + 6, 170, "red" if do < -0.01 else "grey", "δ = " + v, False, 34, 17)
    o += txt(14, 324, 532, t("只有 (0, 0) 的輸出離目標遠，δ 最大；其餘三筆都很小。", "Only (0, 0) is far from its target, so its δ is the largest; the other three are tiny."), 16, TXT, False, "left", 48)
    o += card(14, 380, 532, 84, "yellow", .16) + formula(14, 392, 532, r"J:\ %.5f\ \to\ %.5f\quad(\eta=1.5)" % (J0, J1), 24, 58)
    o += cardH(576, 0, 584, 480, "green", t("更新量 Δw = (η/N) Σ δ · 輸入：九格，顏色越深越大", "Update Δw = (η/N) Σ δ · input: nine cells, darker means larger"))
    mx = float(np.abs(dW).max()); lab = [t("隱藏 1", "hidden 1"), t("隱藏 2", "hidden 2"), t("輸出", "output")]; cl = [t("偏置", "bias"), t("輸入 1", "input 1"), t("輸入 2", "input 2")]; cl2 = [t("偏置", "bias"), "h₁", "h₂"]
    for j in range(3): o += txt(700 + j * 150, 48, 140, (cl2 if False else cl)[j] + (" / h" + str(j) if False else ""), 16, MUT, True, "center", 24)
    for i in range(3):
        o += txt(584, 108 + i * 100, 110, lab[i], 17, TXT, True, "center", 28)
        for j in range(3):
            v = dW[i, j]; a = min(1, abs(v) / mx); x0 = 700 + j * 150; y0 = 78 + i * 100
            o += f'<div style="position:absolute;left:{x0}px;top:{y0}px;width:140px;height:88px;border-radius:12px;background:rgba(217,119,87,{0.12 + 0.78 * math.sqrt(a):.2f});border:2px solid rgba(217,119,87,0.6)"></div>'
            o += txt(x0, y0 + 26, 140, ("0" if abs(v) < 5e-6 else (f"{v:.4f}" if abs(v) >= 5e-5 else f"{v:.5f}")).replace("-", "−"), 19, TXT, True, "center", 30)
    o += txt(700, 372, 450, t("輸出層的 j = 1、2 欄是 h₁、h₂ 的權重；隱藏層的 1、2 欄是 x₁、x₂ 的權重。", "In the output row the last two columns are the weights from h₁ and h₂; in hidden rows they are from x₁ and x₂."), 16, MUT, False, "left", 48)
    o += card(590, 424, 558, 44, "yellow", .16) + txt(598, 430, 542, t("最深的兩格：輸出偏置 −0.0183、隱藏 1 偏置 −0.0155。", "Darkest: output bias −0.0183, hidden-1 bias −0.0155."), 17, TXT, True, "center", 32)
    return o

# ---------------------------------------------------- 4: J curve
def logx(e): return math.log10(max(e, 1))
def page4(lang):
    t = tr(lang); d = D.data(); h, e, Wf = d["main"]; o = ""
    o += cardH(0, 0, 760, 480, "teal", t("J 對 epoch（η = 1.5，批次訓練，兩軸皆對數）", "J against epoch (η = 1.5, batch training, both axes log)"))
    ax = Axes(100, 56, 630, 330, (0, 3.8), (-3.9, -1.9), log=True); ax.frame([0, 1, 2, 3], [1e-2, 1e-3, 1e-4], yfmt=lambda v: "10" + _sup(int(round(math.log10(v)))), xnames={0: "1", 1: "10", 2: "100", 3: "1000"})
    ax.line([(logx(k), float(h[k])) for k in sorted(set([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 3500, e]))], TE, 4)
    ax.line([(0, 2e-4), (3.8, 2e-4)], RD, 2, "9 7")
    for k in (1, 10, 100, 1000, e): ax.dot(logx(k), float(h[k]), TE if k != e else YE, 8)
    ax.text(logx(1000), float(h[1000]), "0.00046", TXT, "start", 16, True, dx=12, dy=-12).text(logx(100), float(h[100]), "0.00175", TXT, "start", 16, True, dx=12, dy=-12).text(logx(e), 2e-4, str(e), "#e0c85f", "middle", 17, True, dy=-16)
    ax.text(0.0, 2e-4, t("停止門檻 0.0002", "stop bar 0.0002"), "#e08a7a", "start", 16, True, dx=8, dy=22)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("epoch（對數）", "epoch (log)")}</text>')
    o += ax.html()
    o += txt(14, 440, 732, t("J 一路緩降；第 %d 個 epoch 才第一次低於 0.0002。" % e, "J falls slowly; only at epoch %d does it first drop below 0.0002." % e), 16, TXT, False, "center", 24)
    o += cardH(776, 0, 384, 480, "orange", t("停止條件與 J 的數值", "The stop rule and the values of J"))
    o += named(790, 46, 356, t("停止條件：代價 J 小於門檻，或梯度小到幾乎不變。本例取 J < 0.0002 且四點全對。", "Stop rule: the cost J falls below a bar, or the gradient is so small that nothing changes. Here J < 0.0002 and all four points correct."), "term:" + t("停止條件", "Stop rule"), 16, TXT, 120)
    for i, k in enumerate((1, 10, 100, 1000)):
        y0 = 184 + i * 40; o += card(790, y0, 356, 34, "grey", .1) + txt(796, y0 + 4, 120, f"epoch {k}", 16, TXT, True, "left", 26) + txt(930, y0 + 4, 210, f"J = {h[k]:.5f}", 16, TXT, False, "right", 26)
    o += card(790, 350, 356, 40, "yellow", .16) + txt(796, 356, 344, f"epoch {e}：J < 0.0002", 17, TXT, True, "center", 28)
    outs = [float(x) for x in N.evalJ(Wf)[2]]
    o += txt(790, 404, 356, t("此時四個輸出：%.3f、%.3f、%.3f、%.3f（依 (0,0)、(0,1)、(1,0)、(1,1) 的順序）〔程式計算〕" % (outs[0], outs[1], outs[2], outs[3]), "The four outputs now: %.3f, %.3f, %.3f, %.3f (in the order (0,0), (0,1), (1,0), (1,1)) (computed)" % (outs[0], outs[1], outs[2], outs[3])), 16, MUT, False, "left", 72)
    return o

# ---------------------------------------------------- 5: learning rate dilemma
def page5(lang):
    t = tr(lang); d = D.data()["eta"]; o = ""
    o += cardH(0, 0, 760, 480, "orange", t("所需 epoch 數隨學習率 η（對數座標）", "Epochs needed against the learning rate η (log scale)"))
    ax = Axes(100, 56, 630, 320, (0, 8), (1, 5)); ax.raw(f'<line x1="{ax.x}" y1="{ax.y}" x2="{ax.x}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/><line x1="{ax.x}" y1="{ax.y+ax.h}" x2="{ax.x+ax.w}" y2="{ax.y+ax.h}" stroke="#b7b0a5" stroke-width="2"/>')
    for e_ in (1, 2, 3, 4, 5): py = ax.Y(e_); ax.raw(f'<line x1="{ax.x}" y1="{py:.1f}" x2="{ax.x+ax.w}" y2="{py:.1f}" stroke="#6b645a" stroke-width="1" opacity="0.45"/><text x="{ax.x-8}" y="{py+5:.1f}" text-anchor="end" font-size="16" fill="#b7b0a5">10{_sup(e_)}</text>')
    o_ = ""; gw = ax.w / 8
    for i, (eta, v) in enumerate(d.items()):
        bx = ax.x + i * gw + 10; bw = gw - 20
        if v:
            top = ax.Y(math.log10(v)); o_ += box(bx, top, bw, ax.y + ax.h - top, "#e8a56a", "#a8452b", 180, shadow=False, r=8) + txt(bx - 12, top - 30, bw + 24, f"{v:,}", 16, TXT, True, "center", 26)
        else:
            top = ax.Y(4.6); o_ += box(bx, top, bw, ax.y + ax.h - top, "#e07a70", "#7a2f2a", 180, shadow=False, r=8) + txt(bx - 4, top + 70, bw + 8, t("失敗", "failed"), 17, TXT, True, "center", 26)
        o_ += txt(bx - 10, ax.y + ax.h + 8, bw + 20, f"{eta:g}", 17, TXT, True, "center", 26)
    o += ax.html() + o_ + txt(100, 420, 630, t("學習率 η", "learning rate η"), 16, MUT, True, "center", 24)
    o += cardH(776, 0, 384, 480, "red", t("兩難：太小太慢、太大失敗", "The dilemma: too small is slow, too large fails"))
    o += chip(790, 52, 356, "blue", t("η = 0.1：55,607 個 epoch", "η = 0.1: 55,607 epochs"), False, 38, 17)
    o += chip(790, 98, 356, "green", t("η = 100：53 個 epoch", "η = 100: 53 epochs"), False, 38, 17)
    o += card(790, 148, 356, 126, "red", .2) + txt(798, 154, 340, t("η = 200：J 前兩步就變大（0.00837 → 0.0828 → 0.25），四個輸出都是 1.00，只對 2 點，訓練失敗。", "η = 200: J grows in the first two steps (0.00837 → 0.0828 → 0.25), all four outputs are 1.00 and only 2 points are right: training fails."), 16, TXT, True, "left", 110)
    o += txt(790, 286, 356, t("來源：最佳 η 取決於代價在權重空間的形狀；平坦的極小值可大步走，陡峭狹窄的要小步。", "The source: the best η depends on the shape of the cost in weight space; flat minima allow big steps, steep narrow ones need small steps."), 16, MUT, False, "left", 96)
    o += card(790, 392, 356, 72, "yellow", .16) + txt(798, 398, 340, t("迷思：越大越好。其實是在某個範圍內越大越快。", "Myth: larger is better. Really, larger is faster only within some range."), 17, TXT, True, "center", 56)
    return o

# ---------------------------------------------------- 6: zero init
def page6(lang):
    t = tr(lang); d = D.data(); h, e, _ = d["main"]; z = d["zero"]; o = ""
    o += cardH(0, 0, 640, 480, "violet", t("全 0 起始值：J 一動也不動", "All-zero start: J does not move at all"))
    ax = Axes(100, 56, 510, 330, (0, 3.6), (-3.9, -0.5), log=True); ax.frame([0, 1, 2, 3], [1e-1, 1e-2, 1e-3, 1e-4], yfmt=lambda v: {0.1: "0.1"}.get(v, "10" + _sup(int(round(math.log10(v))))), xnames={0: "1", 1: "10", 2: "100", 3: "1000"})
    ax.line([(logx(k), float(z[k])) for k in (1, 3, 10, 30, 100, 300, 1000, 3000, 3707)], RD, 4).line([(logx(k), float(h[k])) for k in (1, 3, 10, 30, 100, 300, 1000, 3000, 3707)], TE, 4)
    ax.dot(0, float(z[1]), RD, 7).dot(0, float(h[1]), TE, 7)
    ax.text(1.8, float(z[1]), "J = 0.125", "#e08a7a", "middle", 17, True, dy=-16)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("epoch（對數）", "epoch (log)")}</text>')
    o += ax.html() + legend([(RD, t("全 0 起始值", "all-zero start")), (TE, t("來源起始值", "source start"))], 360, 330, 220)
    o += txt(14, 440, 612, t("全 0 的曲線平躺在 0.125，一個 epoch 也沒降。", "The all-zero curve stays flat at 0.125; it never drops."), 16, TXT, False, "center", 24)
    o += cardH(656, 0, 504, 480, "orange", t("為什麼全 0 動不了", "Why all zeros cannot move"))
    o += named(668, 46, 480, t("初始化：訓練前先給每個權重一個起始值，通常是小的隨機值（來自偽隨機數產生器）。", "Initialization: give every weight a starting value before training, usually small and random (from a pseudo-random generator)."), "term:" + t("初始化", "Initialization"), 16, TXT, 76)
    o += txt(668, 132, 480, t("全 0：每個輸出都是 0.5，輸出層的 δ 為", "All zeros: every output is 0.5 and the output δ is"), 16, TXT, False, "left", 26)
    o += chip(668, 164, 232, "red", "(0 − 0.5)·0.25 = −0.125", False, 36, 16) + chip(916, 164, 232, "blue", "(1 − 0.5)·0.25 = +0.125", False, 36, 16)
    o += txt(668, 212, 480, t("四筆的 δ 正負抵銷，總和為 0；隱藏層的 δ 又乘輸出權重 0，也是 0。", "The four δ cancel to a sum of 0, and the hidden δ is multiplied by the output weight 0, so it is 0 too."), 16, TXT, False, "left", 72)
    o += card(668, 296, 480, 56, "yellow", .16) + txt(676, 302, 464, t("所有梯度剛好是 0：J = 0.125 一動也不動。", "Every gradient is exactly 0: J = 0.125 and nothing moves."), 17, TXT, True, "center", 44)
    lo, hi, n = d["small20"]
    o += txt(668, 366, 480, t("改用 20 組小隨機起點（每個權重在 ±0.5 均勻取），η = 1.5：", "With 20 small random starts (each weight uniform in ±0.5), η = 1.5:"), 16, TXT, False, "left", 48)
    o += chip(668, 416, 480, "green", t(f"全部 {n} 組都達到 J < 0.0002，用了 {lo:,}–{hi:,} 個 epoch", f"all {n} reach J < 0.0002, using {lo:,} to {hi:,} epochs"), False, 40, 16)
    return o

# ---------------------------------------------------- 7: local minimum
def page7(lang):
    t = tr(lang); d = D.data(); A = d["A"]; Bh, Be = d["B"]; o = ""
    o += cardH(0, 0, 700, 480, "red", t("兩個起點：A 卡住、B 走到終點（η = 3）", "Two starts: A gets stuck, B reaches the end (η = 3)"))
    ax = Axes(100, 56, 570, 330, (0, 4.4), (-3.9, -0.3), log=True); ax.frame([0, 1, 2, 3, 4], [1e-1, 1e-2, 1e-3, 1e-4], yfmt=lambda v: {0.1: "0.1"}.get(v, "10" + _sup(int(round(math.log10(v))))), xnames={0: "1", 1: "10", 2: "100", 3: "1000", 4: "10⁴"})
    ks = [1, 3, 10, 30, 100, 300, 1000, 2000, 4000, 6000, 8000]
    ax.line([(logx(k), float(A[k])) for k in ks + [10000, 15000, 20000]], RD, 4).line([(logx(k), float(Bh[k])) for k in ks + [Be]], TE, 4)
    ax.line([(0, 2e-4), (4.4, 2e-4)], MUT, 2, "9 7")
    ax.dot(logx(20000), float(A[20000]), RD, 8).dot(logx(Be), float(Bh[Be]), YE, 8)
    ax.text(logx(20000), float(A[20000]), "J = 0.0626", "#e08a7a", "end", 17, True, dx=-10, dy=-16).text(logx(Be), 2e-4, str(Be), "#e0c85f", "end", 17, True, dx=-6, dy=-14)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+50}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("epoch（對數）", "epoch (log)")}</text>')
    o += ax.html() + legend([(RD, t("起點 A（種子 0）", "start A (seed 0)")), (TE, t("起點 B（種子 4）", "start B (seed 4)"))], 140, 300, 260)
    o += txt(14, 440, 672, t("虛線是停止門檻 0.0002；B 在第 %d 個 epoch 穿過。" % Be, "The dashed line is the stop bar 0.0002; B crosses it at epoch %d." % Be), 16, TXT, False, "center", 24)
    o += cardH(716, 0, 444, 480, "orange", t("區域極小值", "Local minimum"))
    o += named(728, 46, 420, t("區域極小值：附近最低、但不是最低的點；梯度為 0，梯度下降走進去就出不來。", "Local minimum: the lowest point nearby but not the lowest overall; the gradient is 0 there, so gradient descent cannot leave."), "term:" + t("區域極小值", "Local minimum"), 16, TXT, 100)
    o += txt(728, 156, 420, t("300 組隨機起點（種子 0–299，每個權重在 ±3 均勻取），η = 3、最多 20000 個 epoch：", "300 random starts (seeds 0–299, each weight uniform in ±3), η = 3, at most 20000 epochs:"), 16, TXT, False, "left", 72)
    n = d["seeds300"]; o += chip(728, 236, 202, "green", t(f"{n} 組達標", f"{n} reached it"), True, 40, 18) + chip(946, 236, 202, "red", t(f"{300-n} 組沒有", f"{300-n} did not"), True, 40, 18)
    o += card(728, 292, 420, 172, "red", .14) + txt(736, 298, 404, t("起點 A：輸出 0.01、0.99、0.5、0.5，梯度長度只有 %.1f×10⁻⁵；再多跑 6 萬個 epoch，J 仍是 %.4f。" % (d["A_grad"] * 1e5, d["A_more"]), "Start A: outputs 0.01, 0.99, 0.5, 0.5, gradient length only %.1f×10⁻⁵; after 60,000 more epochs J is still %.4f." % (d["A_grad"] * 1e5, d["A_more"])), 16, TXT, False, "left", 150)
    return o

# ---------------------------------------------------- 8: batch vs pattern
def page8(lang):
    t = tr(lang); d = D.data(); o = ""
    items = [(t("逐筆 η = 1.5", "pattern η = 1.5"), d["pat"], ("#5fb88c", "#2c6140")), (t("批次 η = 1.5", "batch η = 1.5"), 3707, ("#d97757", "#94503a")), (t("批次 η = 6", "batch η = 6"), d["batch6"], ("#6a9fd6", "#2f4f7a"))]
    o += cardH(0, 0, 640, 480, "blue", t("三種做法所需 epoch 數（來源起始權重）", "Epochs needed by three methods (source start weights)"))
    for i, (name, v, (c1, c2)) in enumerate(items):
        y0 = 70 + i * 96; o += txt(14, y0 + 6, 160, name, 18, TXT, True, "left", 30)
        bw = v / 3707 * 340; o += box(180, y0, bw, 54, c1, c2, 90, shadow=False, r=8) + txt(180 + bw + 10, y0 + 10, 100, f"{v:,}", 22, TXT, True, "left", 32)
    o += txt(14, 372, 612, t("逐筆與批次 η = 6 幾乎一樣（924 對 926）：逐筆快主要是步長大，不是魔法。", "Pattern mode and batch η = 6 are nearly equal (924 vs 926): pattern mode is fast mainly because of the step size, not magic."), 17, TXT, False, "center", 72)
    o += txt(14, 438, 612, t("三者的總步長相同。〔程式計算〕", "All three have the same total step length. (computed)"), 16, MUT, False, "center", 24)
    o += cardH(656, 0, 504, 480, "orange", t("逐筆訓練", "Pattern training"))
    o += named(668, 46, 480, t("逐筆訓練：每看一筆就用那一筆的 δ 更新一次；隨機性大，來源說有機會跳出區域極小值。", "Pattern (online) training: update after every single sample using that sample's δ; it is noisy, and the source says it may escape a local minimum."), "term:" + t("逐筆訓練", "Pattern training"), 16, TXT, 100)
    o += card(668, 156, 480, 72, "orange", .2) + formula(668, 166, 480, r"w\leftarrow w+\eta\,\delta\,y^{r-1}", 24, 50)
    o += card(668, 244, 480, 150, "red", .14) + txt(676, 250, 464, t("卡住的起點 A：逐筆訓練（固定順序或每個 epoch 洗牌）20000 個 epoch 後仍在 J ≈ 0.063，沒有跳出去。", "The stuck start A: with pattern training (fixed order or reshuffled each epoch) it is still at J ≈ 0.063 after 20000 epochs; it did not escape."), 17, TXT, True, "left", 130)
    o += card(668, 408, 480, 56, "yellow", .16) + txt(676, 414, 464, t("「有機會」不是保證。", "“May escape” is not a guarantee."), 18, TXT, True, "center", 40)
    return o

# ---------------------------------------------------- 9: frozen weights
def page9(lang):
    t = tr(lang); d = D.data(); h, e, Wf = d["main"]; outs = [float(x) for x in N.evalJ(Wf)[2]]; o = ""
    o += cardH(0, 0, 540, 480, "teal", t("訓練完成後凍結的 9 個權重", "The 9 weights frozen after training"))
    o += net(10, 50, Wf)
    o += txt(14, 446, 512, t("第 %d 個 epoch 的權重；之後不再更新。" % e, "The weights at epoch %d; they are never updated again." % e), 16, MUT, False, "center", 24)
    o += cardH(556, 0, 604, 480, "green", t("之後分類只做乘加與 sigmoid", "Classifying afterwards is only multiply-add and sigmoid"))
    o += named(568, 46, 580, t("凍結權重：訓練完成後權重不再更新；之後分類只做乘加與 sigmoid。", "Frozen weights: after training the weights are no longer updated; classifying then needs only multiply-adds and sigmoids."), "term:" + t("凍結權重", "Frozen weights"), 16, TXT, 56)
    o += chip(568, 120, 284, "blue", t("每個輸入：6 次乘法", "per input: 6 multiplications"), False, 40, 17) + chip(864, 120, 284, "orange", t("與 3 次 sigmoid", "and 3 sigmoids"), False, 40, 17)
    for i, (xy, ou, tg) in enumerate(zip(("(0, 0)", "(0, 1)", "(1, 0)", "(1, 1)"), outs, (0, 1, 1, 0))):
        y0 = 184 + i * 50; good = (ou > .5) == (tg == 1); o += card(568, y0, 580, 42, "green" if good else "red", .18)
        o += txt(574, y0 + 6, 120, xy, 17, TXT, True, "center", 28) + txt(700, y0 + 6, 300, t(f"目標 {tg}", f"target {tg}"), 17, TXT, False, "center", 28) + txt(1000, y0 + 6, 140, f"o = {ou:.4f}", 17, TXT, True, "center", 28)
    o += txt(568, 392, 580, t("同一層的單元彼此獨立，所以可以平行；有光學、VLSI 與專用神經計算機的實作（來源）。", "Units in the same layer are independent, so they can run in parallel; optical, VLSI and dedicated neural-computer implementations exist (source)."), 16, TXT, False, "left", 72)
    return o

# ---------------------------------------------------- 10: exercises
def page10(lang):
    t = tr(lang); o = ""; ho = N.fwd(N.W0, np.array([0., 0.]))[0]; ou = N.fwd(N.W0, np.array([0., 0.]))[1]; do = (0 - ou) * ou * (1 - ou)
    c1 = 1.5 * do
    qs = [("blue", t("題 1", "Q1"), t("起始權重，逐筆模式只看 (0, 0)（目標 0），η = 1.5。前向已知 h₁ = 0.1192、h₂ = 0.0009、o = 0.2578。輸出層的 δ 是多少？", "Start weights, pattern mode, only (0, 0) (target 0), η = 1.5; forward h₁ = 0.1192, h₂ = 0.0009, o = 0.2578. Output δ?"),
           [r"\delta_o=(t-o)\,o\,(1-o)", r"=(0-0.2578)\times0.2578\times0.7422=-0.0493"], t("輸出層 δ = −0.0493。", "output-layer δ = −0.0493."), t("核對：輸出 0.2578 比目標 0 高，所以 δ 是負的。", "Check: the output 0.2578 is above the target 0, so δ is negative.")),
          ("red", t("題 2", "Q2"), t("輸出層三個權重的更新量是多少？（Δw = η · δ · 輸入）", "What are the updates of the three output weights? (Δw = η · δ · input)"),
           [r"\Delta w=1.5\times(-0.0493)\times\text{input}", r"\text{bias}:-0.0740\quad w_1:-0.0088\quad w_2:-0.0001"], t("偏置 −0.0740、來自 h₁ 的權重 −0.0088、來自 h₂ 的 −0.0001。", "bias −0.0740, weight from h₁ −0.0088, weight from h₂ −0.0001."), t("核對：三個都是負的，輸出層權重都往下調；隱藏單元 1 的 δ = −0.0414，偏置 −0.0622。", "Check: all three are negative, so the output weights move down; hidden unit 1 has δ = −0.0414 and bias −0.0622."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 50, 538, q, 17, TXT, True, "left", 96)
        o += chip(x + 16, 156, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 196, 538, 150, "grey", .1) + formula(x + 16, 204, 538, tex[0], 21 if i == 0 else 20, 56) + formula(x + 16, 272, 538, tex[1], 19 if i == 0 else 18, 62)
        o += chip(x + 16, 360, 538, role, ans, True, 42, 17)
        o += txt(x + 16, 414, 538, chk, 16, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 11)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp07_" + fid, fn, W, H, out)
