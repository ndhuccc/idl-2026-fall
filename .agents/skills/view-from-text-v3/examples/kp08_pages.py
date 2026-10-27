"""KP08 pages designed in HTML (view-from-text-v3, route A). pageN(lang) -> body HTML of a 1160x480 figure."""
import os, sys, math
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
import kp08_nums as N
W, H = 1160, 480
def box(x, y, w, h, c1, c2, ang=135, shadow=True, r=14):
    return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;border-radius:{r}px;background:linear-gradient({ang}deg,{c1},{c2});' + ('box-shadow:0 10px 18px 0 rgba(0,0,0,0.45);' if shadow else '') + '"></div>'
def cardH(x, y, w, h, role, title, size=17): return card(x, y, w, h, role) + head(x, y, w, role, title, 36, size)
def sg(v, d=3):
    """signed number with a true minus sign"""
    s = f"{v:.{d}f}".rstrip("0").rstrip("."); s = "0" if s in ("-0", "") else s
    return s.replace("-", "−")
def f4(a): return "(" + ", ".join(sg(float(v), 4) for v in a) + ")"
def f2(a): return "(" + ", ".join(sg(float(v)) for v in a) + ")"
RUN = 0.6

def valley(ax, levels=(1.5, 3, 4.5, 6)):
    """contour ellipses of J = 1/2 (w1^2 + 20 w2^2) and the axes through the origin"""
    ax.raw(f'<rect x="{ax.x}" y="{ax.y}" width="{ax.w}" height="{ax.h}" fill="none" stroke="#6b645a" stroke-width="2"/>')
    ax.raw(f'<line x1="{ax.X(ax.xr[0])}" y1="{ax.Y(0)}" x2="{ax.X(ax.xr[1])}" y2="{ax.Y(0)}" stroke="#6b645a" stroke-width="1.5"/><line x1="{ax.X(0)}" y1="{ax.Y(ax.yr[0])}" x2="{ax.X(0)}" y2="{ax.Y(ax.yr[1])}" stroke="#6b645a" stroke-width="1.5"/>')
    for a in levels:
        b = a / math.sqrt(20); pts = " ".join(f"{ax.X(a*math.cos(th)):.1f},{ax.Y(b*math.sin(th)):.1f}" for th in np.linspace(0, 2 * math.pi, 25)[:-1])
        ax.raw(f'<polygon points="{pts}" fill="#5b7fa8" fill-opacity="0.07" data-smooth="all"/>')
        ax.line([(a * math.cos(th), b * math.sin(th)) for th in np.linspace(0, 2 * math.pi, 49)], "#5b7fa8", 2)
def path(ax, p, col, w=3, dots=18, r=5):
    ax.line([(float(a), float(b)) for a, b in p], col, w)
    for a, b in p[:dots + 1]: ax.dot(float(a), float(b), col, r)
    return ax

# ------------------------------------------------ 1: narrow valley
def page1(lang):
    t = tr(lang); n, p = N.plain(); o = ""
    o += cardH(0, 0, 760, 480, "blue", t("等高線 J = ½(w₁² + 20w₂²) 與普通梯度下降的軌跡", "Contours of J = ½(w₁² + 20w₂²) and the plain gradient-descent path"))
    ax = Axes(44, 50, 690, 330, (-6.6, 6.6), (-1.7, 1.7)); valley(ax); path(ax, p, RD, 3, 40, 4)
    ax.dot(-4, 1, YE, 8).dot(0, 0, TXT, 6)
    ax.text(-4, 1, t("起點 (−4, 1)", "start (−4, 1)"), YE, "start", 16, True, dx=12, dy=-12).text(0, 0, t("最小值 (0, 0)", "minimum (0, 0)"), TXT, "start", 16, True, dx=10, dy=-14)
    ax.text(6.4, 0, "w₁", MUT, "end", 16, True, dy=-10).text(0, 1.62, "w₂", MUT, "start", 16, True, dx=8, dy=2)
    o += ax.html()
    o += chip(44, 392, 20, "red", "", False, 6) if False else ""
    o += legend([(RD, t("普通梯度下降（μ = 0.08），共 %d 步" % n, "plain gradient descent (μ = 0.08), %d steps" % n))], 56, 396, 600)
    o += txt(20, 428, 720, t("藍色橢圓是等高線（J = 1.125、4.5、10.125、18），越扁表示兩個方向的彎曲程度差越多。", "Blue ellipses are contours (J = 1.125, 4.5, 10.125, 18); the flatter, the bigger the gap in curvature between the two directions."), 16, MUT, False, "left", 48)
    o += cardH(776, 0, 384, 480, "orange", t("名詞", "Terms"))
    o += named(792, 50, 352, t("曲率：代價面彎得多厲害；某方向曲率大，沿那個方向走一小步，代價就變很多。", "Curvature: how sharply the cost surface bends; where curvature is large, one small step changes the cost a lot."), "term:" + t("曲率", "Curvature"), 16, TXT, 120)
    o += named(792, 186, 352, t("狹長谷底（病態代價面）：一個方向曲率很大、另一個很小，等高線是很扁的橢圓。", "Narrow valley (ill-conditioned surface): very large curvature in one direction and very small in the other, so the contours are flat ellipses."), "term:" + t("狹長谷底", "Narrow valley"), 16, TXT, 140)
    o += card(792, 348, 352, 116, "yellow", .14) + txt(800, 354, 336, t("w₂ 方向曲率 20、w₁ 方向曲率 1；起點 (−4, 1)、μ = 0.08：走 100 步才到 |w| < 0.001〔程式計算〕", "Curvature 20 along w₂, 1 along w₁; start (−4, 1), μ = 0.08: 100 steps to reach |w| < 0.001 (computed)"), 16, TXT, True, "center", 104)
    return o

# ------------------------------------------------ 2: why it zigzags
def page2(lang):
    t = tr(lang); o = ""; T = 9
    def stem(x0, y0, col, vals, name):
        ax = Axes(x0 + 56, y0, 480, 118, (-0.5, T - 0.5), (-1.25, 1.25) if name == "w2" else (-4.6, 0.6)); ax.frame(range(T), [], xnames={})
        zero = ax.Y(0); ax.raw(f'<line x1="{ax.x}" y1="{zero:.1f}" x2="{ax.x+ax.w}" y2="{zero:.1f}" stroke="#6b645a" stroke-width="1.5"/>')
        pts = list(enumerate(vals)); ax.line(pts, col, 3)
        for i, v in pts: ax.dot(i, v, col, 6)
        for i, v in pts[:5]:
            up = (v >= 0)
            ax.text(i, v, sg(v, 3), TXT, "middle", 16, True, dy=(-14 if up else 28) if name == "w2" else (-14 if i % 2 == 0 else -38))
        return ax.html()
    v2 = [1.0 * (-0.6) ** i for i in range(T)]; v1 = [-4.0 * 0.92 ** i for i in range(T)]
    o += cardH(0, 0, 572, 316, "red", t("陡的方向 w₂：每步變號、只縮到 0.6 倍", "Steep direction w₂: the sign flips every step, only 0.6 times smaller"))
    o += formula(8, 40, 556, r"w_2\leftarrow w_2-0.08\cdot 20\,w_2=-0.6\,w_2", 21, 42)
    o += stem(0, 96, RD, v2, "w2") + txt(60, 240, 480, t("步數 t（0 是起點）", "step t (0 is the start)"), 16, MUT, False, "center", 24)
    o += txt(14, 262, 544, t("1 → −0.6 → 0.36 → −0.216：上下擺，每回只縮 40%", "1 → −0.6 → 0.36 → −0.216: it swings up and down, shrinking only 40% each time"), 16, TXT, False, "center", 48)
    o += cardH(588, 0, 572, 316, "blue", t("平的方向 w₁：同方向、每步只縮 8%", "Flat direction w₁: same direction, only 8% smaller per step"))
    o += formula(596, 40, 556, r"w_1\leftarrow w_1-0.08\cdot 1\cdot w_1=0.92\,w_1", 21, 42)
    o += stem(588, 96, BL, v1, "w1") + txt(648, 240, 480, t("步數 t（0 是起點）", "step t (0 is the start)"), 16, MUT, False, "center", 24)
    o += txt(602, 262, 544, t("−4 → −3.68 → −3.386 → −3.115：一路靠近 0，但很慢", "−4 → −3.68 → −3.386 → −3.115: steadily toward 0, but slowly"), 16, TXT, False, "center", 48)
    o += cardH(0, 332, 1160, 148, "yellow", t("兩難：μ 要小於 2/20 = 0.1，w₂ 才不會越擺越大", "Dilemma: μ must stay below 2/20 = 0.1 so that w₂ does not swing wider and wider"))
    o += formula(14, 378, 540, r"|1-\mu\cdot 20|<1\ \Rightarrow\ \mu<\frac{2}{20}=0.1", 22, 52)
    o += chip(580, 380, 566, "red", t("μ = 0.12：因子 1 − 2.4 = −1.4，60 步後 |w| = 5.86×10⁸〔程式計算〕", "μ = 0.12: factor 1 − 2.4 = −1.4, |w| = 5.86×10⁸ after 60 steps (computed)"), False, 40, 16)
    o += txt(580, 428, 566, t("但這麼小的步長讓 w₁ 走得慢：問題出在兩個方向要不同的步長。", "But so small a step makes w₁ crawl: the two directions want different step sizes."), 16, TXT, False, "left", 44)
    return o

# ------------------------------------------------ 3: momentum definition + hand calculation
def page3(lang):
    t = tr(lang); o = ""; n, p, ds = N.momentum(RUN)
    o += cardH(0, 0, 450, 480, "violet", t("動量：更新帶著慣性", "Momentum: updates carry inertia"))
    o += card(14, 52, 422, 90, "violet", .22) + formula(14, 62, 422, r"\Delta w(t)=\alpha\,\Delta w(t-1)-\mu\,g(t)", 24, 60)
    o += chip(14, 156, 202, "blue", t("慣性：上一步的 α 倍", "inertia: α × last step"), False, 40, 16) + chip(234, 156, 202, "orange", t("本次梯度步 −μ g", "this gradient step −μg"), False, 40, 16)
    o += named(20, 214, 410, t("動量：讓更新帶著慣性，這一步除了本次的梯度步，還加上上一步的 α 倍；α 叫動量因子，實務上取 0.1 到 0.8。", "Momentum: let the update carry inertia; besides this gradient step, add α times the previous update. α is the momentum factor, 0.1 to 0.8 in practice."), "term:" + t("動量", "Momentum"), 16, TXT, 130)
    o += card(14, 360, 422, 104, "yellow", .14) + txt(22, 368, 406, t("α = 0 就是普通梯度下降；本頁 μ = 0.08、α = 0.6、起點 w = (−4, 1)。", "α = 0 is plain gradient descent; here μ = 0.08, α = 0.6, start w = (−4, 1)."), 16, TXT, True, "center", 88)
    o += cardH(466, 0, 694, 480, "green", t("手算前三步（μ = 0.08、α = 0.6）", "The first three steps by hand (μ = 0.08, α = 0.6)"))
    xs = [(474, 50, "t"), (528, 150, t("梯度 g", "gradient g")), (684, 170, t("更新量 Δw", "update Δw")), (860, 170, t("更新後 w", "new w"))]
    for xx, ww, lab in xs: o += txt(xx, 46, ww, lab, 16, MUT, True, "center", 24)
    for i in range(3):
        g, d = ds[i]; y0 = 76 + i * 50; hot = i == 1; o += card(474, y0, 676, 42, "green" if hot else "grey", .22 if hot else .1)
        o += txt(474, y0 + 6, 50, str(i + 1), 17, TXT, True, "center", 28) + txt(528, y0 + 6, 150, f2(g), 17, TXT, False, "center", 28) + txt(684, y0 + 6, 170, f4(d), 17, TXT, True, "center", 28) + txt(860, y0 + 6, 170, f2(p[i + 1]), 17, TXT, False, "center", 28)
    o += txt(474, 232, 676, t("第 2 步的算式（Δw(1) = (0.32, −1.6)）：", "Step 2 worked out (Δw(1) = (0.32, −1.6)):"), 17, TXT, True, "left", 28)
    o += card(474, 266, 676, 70, "grey", .1) + formula(474, 274, 676, r"0.6\times(0.32,-1.6)-0.08\times(-3.68,-12)=(0.4864,\ 0)", 22, 54)
    o += chip(474, 352, 330, "blue", t("w₂：慣性 0.6 × (−1.6) = −0.96", "w₂: inertia 0.6 × (−1.6) = −0.96"), False, 38, 16) + chip(820, 352, 330, "orange", t("w₂：梯度 −0.08 × (−12) = +0.96", "w₂: gradient −0.08 × (−12) = +0.96"), False, 38, 16)
    o += card(474, 404, 676, 60, "yellow", .16) + txt(482, 410, 660, t("−0.96 + 0.96 = 0：兩股力量剛好抵銷，w₂ 這一步不動，w₁ 的步子卻變大了。", "−0.96 + 0.96 = 0: the two forces cancel, w₂ stays put while the w₁ step grows."), 17, TXT, True, "center", 50)
    return o

# ------------------------------------------------ 4: exponential weighted average
def page4(lang):
    t = tr(lang); o = ""; T = 6
    o += cardH(0, 0, 640, 480, "teal", t("歷史梯度的權重 αᵗ：越久以前越小", "Weights αᵗ on past gradients: the older, the smaller"))
    ax = Axes(66, 70, 540, 280, (0, 1), (0, 1.12)); ax.frame([], [0, .25, .5, .75, 1], yfmt=lambda v: f"{v:g}"); gw = ax.w / T; o_ = ""
    for i in range(T):
        for j, (a, c1, c2) in enumerate(((0.6, "#6a9fd6", "#2f4f7a"), (0.8, "#d9822b", "#a8452b"))):
            v = a ** i; bw = 40; bx = ax.x + i * gw + (gw - 2 * bw) / 2 + j * bw; top = ax.Y(v)
            o_ += box(bx + 3, top, bw - 6, ax.y + ax.h - top, c1, c2, 180, shadow=False, r=7)
            if i > 0 or j == 0: o_ += txt(bx - 8 + (bw / 2 if i == 0 else 0), top - 30, bw + 16, "1" if i == 0 else f"{round(v,3):g}", 16, TXT, True, "center", 26)
        o_ += txt(ax.x + i * gw, ax.y + ax.h + 8, gw, f"t = {i}", 16, MUT, True, "center", 24)
    o += ax.html() + o_
    o += legend([("#6a9fd6", "α = 0.6"), ("#d9822b", "α = 0.8")], 360, 82, 200)
    o += txt(14, 414, 612, t("t = 0 是最近的梯度；每往前一回合，權重再乘 α。", "t = 0 is the latest gradient; each step back multiplies the weight by α."), 16, TXT, False, "center", 48)
    o += cardH(656, 0, 504, 480, "orange", t("一個機制：該快時快、該穩時穩", "One mechanism: fast where flat, steady where steep"))
    o += named(668, 46, 480, t("指數加權平均：最近的梯度權重最大，越久以前的每回合再乘 α 變小，權重是 1、α、α²、…。", "Exponentially weighted average: the latest gradient weighs most and each older one is multiplied by α again, giving weights 1, α, α², …."), "term:" + t("指數加權平均", "Exponentially weighted average"), 16, TXT, 100)
    o += formula(662, 150, 492, r"\Delta w(T)=-\mu\sum_{t}\alpha^{t}g(T-t)+\alpha^{T}\Delta w(0)", 18, 40)
    o += txt(668, 196, 480, t("梯度不變（平的方向）：總權重 1 + α + α² + ⋯ = 1/(1 − α)", "Constant gradient (flat direction): total weight 1 + α + α² + ⋯ = 1/(1 − α)"), 16, TXT, True, "left", 48)
    o += chip(668, 252, 236, "blue", "α = 0.6：1/0.4 = 2.5", True, 38, 18) + chip(916, 252, 232, "orange", "α = 0.8：1/0.2 = 5", True, 38, 18)
    o += txt(668, 300, 480, t("等於把學習率放大 2.5 倍與 5 倍；固定梯度跑 80 步驗證〔程式計算〕。", "That scales the learning rate by 2.5 and 5; verified by 80 steps with a fixed gradient (computed)."), 16, MUT, False, "left", 48)
    o += txt(668, 356, 480, t("梯度正負交替（陡的方向）：權重 +1、−α、+α²、−α³ … 互相抵消。", "Alternating gradient (steep direction): weights +1, −α, +α², −α³ … cancel each other."), 16, TXT, True, "left", 48)
    o += card(668, 412, 480, 54, "yellow", .14) + txt(676, 418, 464, t("平的方向放大、陡的方向抵銷，同一個式子。", "Amplified where flat, cancelled where steep."), 17, TXT, True, "center", 40)
    return o

# ------------------------------------------------ 5: two paths
def page5(lang):
    t = tr(lang); n0, p0 = N.plain(); n1, p1, _ = N.momentum(RUN); o = ""
    o += cardH(0, 0, 780, 480, "blue", t("同起點、同 μ = 0.08：普通 vs 動量（α = 0.6）", "Same start, same μ = 0.08: plain vs momentum (α = 0.6)"))
    ax = Axes(44, 50, 710, 330, (-6.6, 6.6), (-1.7, 1.7)); valley(ax)
    path(ax, p0, RD, 3, 0, 4); path(ax, p1, GR, 4, n1, 6); ax.dot(-4, 1, YE, 8)
    ax.text(-4, 1, t("起點", "start"), YE, "start", 16, True, dx=12, dy=-12); ax.text(6.4, 0, "w₁", MUT, "end", 16, True, dy=-10).text(0, 1.62, "w₂", MUT, "start", 16, True, dx=8, dy=2)
    o += ax.html() + legend([(RD, t("普通：%d 步" % n0, "plain: %d steps" % n0)), (GR, t("動量 α = 0.6：%d 步" % n1, "momentum α = 0.6: %d steps" % n1))], 56, 394, 250)
    o += txt(350, 394, 400, t("紅色在 w₂ 方向上下擺很多回；綠色的擺幅縮得更快，w₁ 方向也走得更快。", "Red swings up and down in w₂ many times; green's swing shrinks faster and w₁ advances faster too."), 16, TXT, False, "left", 72)
    o += cardH(796, 0, 364, 480, "green", t("步數與代價", "Steps and cost"))
    o += txt(808, 50, 340, t("到 |w| < 0.001 所需步數", "Steps to reach |w| < 0.001"), 16, MUT, True, "center", 24)
    o += box(812, 96, 324 * 100 / 100, 40, "#d97757", "#94503a", 90, shadow=False, r=8) + txt(820, 102, 300, t("普通 100 步", "plain 100"), 18, TXT, True, "left", 28)
    o += box(812, 156, 324 * n1 / 100, 40, "#5fb88c", "#2c6140", 90, shadow=False, r=8) + txt(820, 162, 300, t("動量 %d 步" % n1, "momentum %d" % n1), 18, TXT, True, "left", 28)
    o += card(808, 228, 340, 96, "yellow", .14) + txt(816, 234, 324, t("28 / 100 = 0.28：約快 3.6 倍〔程式計算〕", "28 / 100 = 0.28: about 3.6 times faster (computed)"), 17, TXT, True, "center", 84)
    o += txt(808, 340, 340, t("代價：", "The price:"), 17, TXT, True, "left", 26) + txt(808, 370, 340, t("多存一份上一步的更新量，多一個要調的旋鈕 α。", "one more stored vector (the previous update) and one more knob to tune, α."), 17, TXT, False, "left", 96)
    return o

# ------------------------------------------------ 6: best alpha
def page6(lang):
    t = tr(lang); o = ""; al = [0, .3, .5, .6, .7, .8, .9, .95]; st = [N.momentum(a)[0] for a in al]
    o += cardH(0, 0, 760, 480, "orange", t("動量因子 α 與所需步數（μ = 0.08）", "Momentum factor α against the steps needed (μ = 0.08)"))
    ax = Axes(86, 56, 640, 310, (-0.02, 1.0), (0, 270)); ax.frame([0, .2, .4, .6, .8, 1.0], [0, 50, 100, 150, 200, 250], yfmt=lambda v: f"{v:g}")
    ax.rect(0.875, 1.0, 0, 270, RD, .18); ax.line([(-0.02, 100), (1.0, 100)], MUT, 2, "8 6")
    ax.line(list(zip(al, st)), OR, 4)
    for a, s in zip(al, st): ax.dot(a, s, GR if s == 28 else (RD if a >= 0.9 else OR), 8)
    for a, s in zip(al, st): ax.text(a, s, str(s), GR if s == 28 else TXT, "middle", 17, True, dy=-16 if a not in (0.3, 0.7, 0.8) else -16, dx=0 if a not in (0, ) else 14)
    ax.text(0.6, 28, "", TXT); ax.text(0.4, 100, t("普通梯度下降 100 步", "plain descent: 100 steps"), MUT, "middle", 16, True, dy=-10, dx=-16)
    ax.raw(f'<text x="{ax.x+ax.w/2}" y="{ax.y+ax.h+44}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{t("動量因子 α", "momentum factor α")}</text>')
    o += ax.html()
    o += txt(14, 432, 732, t("曲線在 α = 0.6 最低（28 步）；α ≥ 0.9（紅色區）反而比不用動量更慢。", "Lowest at α = 0.6 (28 steps); α ≥ 0.9 (red zone) is slower than no momentum."), 16, TXT, False, "center", 24)
    o += cardH(776, 0, 384, 480, "red", t("常見迷思：α 越大越好", "Myth: a larger α is always better"))
    o += chip(792, 54, 352, "red", t("α = 0.9：137 步 > 100 步", "α = 0.9: 137 steps > 100"), True, 40, 19) + chip(792, 106, 352, "red", t("α = 0.95：247 步", "α = 0.95: 247 steps"), True, 40, 19)
    o += txt(792, 164, 352, t("慣性太大，在谷底衝過頭，像單擺晃很久。", "Too much inertia overshoots the valley floor and swings like a pendulum before settling."), 17, TXT, False, "left", 84)
    o += card(792, 270, 352, 190, "yellow", .14) + txt(800, 278, 336, t("實務上常取 0.1 到 0.8；最佳值看問題，要試。", "In practice 0.1 to 0.8; the best value depends on the problem, so try several."), 17, TXT, False, "left", 60) + txt(800, 366, 336, t("這個例子裡 α = 0.6 最快；來源報告過 2 倍以上的加速。", "Here α = 0.6 is fastest; the source reports speed-ups of 2 times or more."), 17, TXT, False, "left", 60)
    return o

# ------------------------------------------------ 7: adaptive learning rate
def page7(lang):
    t = tr(lang); o = ""; res = {m: N.adaptive(m) for m in (0.02, 0.12, 0.08)}; pl = {m: N.plain(m)[0] for m in (0.02, 0.12, 0.08)}
    o += cardH(0, 0, 560, 480, "violet", t("規則：看代價 J 的前後比值調 μ", "The rule: adjust μ from the ratio of successive costs J"))
    o += named(14, 44, 500, t("自適應學習率：訓練中看代價的前後比值自動調 μ，變好就慢慢加，變壞就大力煞車。", "Adaptive learning rate: adjust μ automatically from the ratio of successive costs: raise it slowly when things improve, brake hard when they worsen."), "term:" + t("自適應學習率", "Adaptive learning rate"), 16, TXT, 100)
    rows = [("green", r"\frac{J(t)}{J(t-1)}<1", r"\mu\leftarrow 1.05\,\mu", t("變好：慢慢加大", "better: grow slowly")), ("red", r"\frac{J(t)}{J(t-1)}>1.04", r"\mu\leftarrow 0.7\,\mu", t("變壞：大力煞車，這一步不採用、動量歸零", "worse: brake hard; reject the step, reset momentum")), ("grey", t("其餘", "otherwise"), r"\mu\leftarrow\mu", t("維持不變", "unchanged"))]
    for i, (role, cond, upd, note) in enumerate(rows):
        y0 = 154 + i * 98; o += card(14, y0, 532, 88, role, .18)
        o += (formula(24, y0 + 8, 230, cond, 20, 50) if cond.startswith("\\") else txt(24, y0 + 14, 230, cond, 19, TXT, True, "center", 30)) + formula(270, y0 + 8, 270, upd, 22, 50)
        o += txt(24, y0 + 56, 516, note, 16, TXT, False, "center", 26)
    o += txt(14, 450, 532, t("典型值 1.05、0.7、1.04（來源）", "Typical values 1.05, 0.7, 1.04 (from the source)"), 16, MUT, False, "center", 24)
    o += cardH(576, 0, 584, 480, "blue", t("起始 μ₀ 不同時所需步數（|w| < 0.001）", "Steps needed for different initial μ₀ (|w| < 0.001)"))
    ax = Axes(630, 60, 500, 290, (0, 1), (0, 450)); ax.frame([], [0, 100, 200, 300, 400], yfmt=lambda v: f"{v:g}"); gw = ax.w / 3; o_ = ""
    for gi, m in enumerate((0.02, 0.12, 0.08)):
        n, rej = res[m][0], res[m][1]; vals = [(pl[m], "#d97757", "#94503a"), (n, "#5fb88c", "#2c6140")]
        for j, (v, c1, c2) in enumerate(vals):
            bw = 62; bx = ax.x + gi * gw + (gw - 2 * bw) / 2 + j * bw; top = ax.Y(v if v else 440)
            if v is None:
                o_ += box(bx + 4, top, bw - 8, ax.y + ax.h - top, "#e07a70", "#7a2f2a", 180, shadow=False, r=8) + txt(bx - 4, top + 60, bw + 8, t("爆開", "blows up"), 16, TXT, True, "center", 24)
            else:
                o_ += box(bx + 4, top, bw - 8, ax.y + ax.h - top, c1, c2, 180, shadow=False, r=8) + txt(bx + 4, top - 30, bw - 8, str(v), 18, TXT, True, "center", 28)
        o_ += txt(ax.x + gi * gw, ax.y + ax.h + 8, gw, f"μ₀ = {m}", 17, TXT, True, "center", 26)
    o += ax.html() + o_ + legend([("#d97757", t("普通", "plain")), ("#5fb88c", t("自適應", "adaptive"))], 1000, 80, 120)
    o += txt(590, 404, 556, t("μ₀ = 0.12：共 95 步，其中 12 步被拒絕後仍收斂；μ₀ = 0.08 本來就接近最佳，自適應沒比普通好多少（93 對 100）。〔程式計算〕", "μ₀ = 0.12: 95 steps in all, 12 of them rejected, yet it converges; μ₀ = 0.08 is already near the best, so adaptive gains little (93 vs 100). (computed)"), 16, TXT, False, "left", 72)
    return o

# ------------------------------------------------ 8: quickprop
def page8(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 590, 480, "teal", t("沿 w₂ 的切面 J = 10w₂²：用兩點斜率估谷底", "Slice along w₂, J = 10w₂²: estimate the floor from two slopes"))
    ax = Axes(60, 60, 500, 320, (-1.3, 1.3), (-0.8, 13.5)); ax.raw(f'<line x1="{ax.x}" y1="{ax.Y(0)}" x2="{ax.x+ax.w}" y2="{ax.Y(0)}" stroke="#6b645a" stroke-width="1.5"/><line x1="{ax.X(0)}" y1="{ax.y}" x2="{ax.X(0)}" y2="{ax.y+ax.h}" stroke="#6b645a" stroke-width="1.5"/>')
    xs = np.linspace(-1.04, 1.04, 45); ax.line([(float(x), float(10 * x * x)) for x in xs], TE, 4)
    for (x0, s_, c) in ((1.0, 20.0, OR), (-0.6, -12.0, VI)):
        y0 = 10 * x0 * x0; d = 0.15 if x0 > 0 else 0.4; ax.line([(x0 - d, y0 - s_ * d), (x0 + d, y0 + s_ * d)] if x0 > 0 else [(x0 - d, y0 - s_ * d), (x0 + d, y0 + s_ * d)], c, 3, "7 5")
    ax.dot(1, 10, OR, 9).dot(-0.6, 3.6, VI, 9).dot(0, 0, GR, 10)
    ax.text(1, 10, t("w₂ = 1，斜率 20", "w₂ = 1, slope 20"), OR, "end", 16, True, dx=-14, dy=6).text(-0.6, 3.6, t("w₂ = −0.6，斜率 −12", "w₂ = −0.6, slope −12"), VI, "end", 16, True, dx=-14, dy=-10).text(0, 0, t("谷底 w₂ = 0", "floor w₂ = 0"), GR, "start", 16, True, dx=16, dy=-16)
    ax.raw(f'<text x="{ax.X(1.28)}" y="{ax.Y(0)-10}" text-anchor="end" font-size="16" font-weight="700" fill="#b7b0a5">w₂</text>')
    o += ax.html() + txt(14, 400, 562, t("第 1 步是普通步（1 → −0.6，Δ = −1.6），第 2 步直接跳到谷底。", "Step 1 is a plain step (1 → −0.6, Δ = −1.6); step 2 jumps straight to the floor."), 16, TXT, False, "center", 48)
    o += cardH(606, 0, 554, 480, "orange", t("quickprop 公式與 w₂ 的跳躍", "The quickprop formula and the w₂ jump"))
    o += named(618, 46, 530, t("quickprop：把每個權重當成各自獨立，用最近兩次的斜率估出拋物線的谷底，直接跳過去。", "quickprop: treat every weight as independent, estimate the parabola's floor from the last two slopes, and jump there."), "term:quickprop", 16, TXT, 76)
    o += card(618, 132, 530, 90, "orange", .2) + formula(618, 136, 530, r"\Delta w(t)=\alpha(t)\,\Delta w(t-1)", 22, 40) + formula(618, 178, 530, r"\alpha(t)=\min\left\{\frac{g(t)}{g(t-1)-g(t)},\ \alpha_{\max}\right\}", 19, 46)
    o += txt(618, 232, 530, t("α 的上限約 1.75（避免一步跳太遠）", "the cap on α is about 1.75 (stops one jump going too far)"), 16, MUT, False, "left", 24)
    o += card(618, 266, 530, 196, "grey", .1) + txt(628, 272, 510, t("例（w₂）：", "Example (w₂):"), 17, TXT, True, "left", 26)
    o += formula(618, 304, 530, r"\alpha=\frac{-12}{20-(-12)}=-0.375", 22, 52) + formula(618, 360, 530, r"\Delta=-0.375\times(-1.6)=+0.6", 22, 44)
    o += chip(634, 410, 498, "green", t("w₂ = −0.6 + 0.6 = 0：第 2 步剛好落在谷底", "w₂ = −0.6 + 0.6 = 0: step 2 lands exactly on the floor"), True, 40, 17)
    return o

# ------------------------------------------------ 9: five methods + Newton
def page9(lang):
    t = tr(lang); o = ""; n_q = N.quickprop()[0]
    items = [(t("普通 μ = 0.02", "plain μ = 0.02"), N.plain(0.02)[0], ("#d97757", "#94503a")), (t("自適應（μ₀ = 0.02）", "adaptive (μ₀ = 0.02)"), N.adaptive(0.02)[0], ("#9a85d6", "#5a4a8a")), (t("普通 μ = 0.08", "plain μ = 0.08"), N.plain()[0], ("#d97757", "#94503a")), (t("動量 α = 0.6", "momentum α = 0.6"), N.momentum(RUN)[0], ("#5fb88c", "#2c6140")), ("quickprop", n_q, ("#d9b84a", "#8a6a1a"))]
    o += cardH(0, 0, 690, 480, "blue", t("五種做法所需步數（|w| < 0.001）", "Steps needed by the five methods (|w| < 0.001)"))
    for i, (name, v, (c1, c2)) in enumerate(items):
        y0 = 62 + i * 72; o += txt(14, y0 + 8, 196, name, 17, TXT, True, "left", 28)
        bw = max(v / 411 * 380, 8); o += box(214, y0, bw, 46, c1, c2, 90, shadow=False, r=8) + txt(214 + bw + 8, y0 + 8, 80, str(v), 20, TXT, True, "left", 30)
    o += txt(14, 428, 662, t("但這個例子每個切面剛好是拋物線，最合 quickprop；真實代價不是二次的。〔程式計算〕", "But here every slice is exactly a parabola, which suits quickprop best; real costs are not quadratic. (computed)"), 16, TXT, False, "center", 48)
    o += cardH(706, 0, 454, 480, "violet", t("延伸：牛頓法", "Extension: Newton's method"))
    o += named(718, 46, 400, t("牛頓法：用曲率資訊（Hessian 矩陣 H，二階偏導數組成）直接算出谷底；二次函數一步就到。", "Newton's method: use curvature (the Hessian matrix H of second derivatives) to compute the floor directly; a quadratic needs one step."), "term:" + t("牛頓法", "Newton's method"), 16, TXT, 100)
    o += card(718, 156, 430, 80, "violet", .2) + formula(718, 168, 430, r"\Delta w=-H^{-1}g", 26, 56)
    o += card(718, 248, 430, 96, "grey", .1) + formula(718, 256, 430, r"H=\mathrm{diag}(1,20)\ \Rightarrow\ \Delta w=-w", 22, 44) + txt(726, 304, 414, t("本例一步到原點", "here: one step to the origin"), 16, TXT, True, "center", 28)
    o += txt(718, 356, 430, t("代價：要算、存、反轉 Hessian。quickprop 可看成每個權重各做一維牛頓法；其他還有共軛梯度、delta-bar-delta。", "Price: compute, store and invert the Hessian. quickprop is a 1-D Newton step per weight; others include conjugate gradient and delta-bar-delta."), 16, TXT, False, "left", 120)
    return o

# ------------------------------------------------ 10: exercises
def page10(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("同一個谷底，μ = 0.08、α = 0.6。第 2 步後 w = (−3.1936, −0.6)，梯度 g = (w₁, 20w₂) 是多少？", "Same valley, μ = 0.08, α = 0.6. After step 2, w = (−3.1936, −0.6): what is g = (w₁, 20w₂)?"), [r"g=(w_1,\ 20\,w_2)", r"=(-3.1936,\ 20\times(-0.6))=(-3.1936,\ -12)"], t("g = (−3.1936, −12)。", "g = (−3.1936, −12)."), t("核對：和前面手算表第 3 步的梯度一致。", "Check: it matches the step-3 gradient of the hand-computed table.")),
          ("red", t("題 2", "Q2"), t("接著第 2 步更新量 Δw(2) = (0.4864, 0)。算 Δw(3) = 0.6 Δw(2) − 0.08 g(3) 與新的 w。", "Next, Δw(2) = (0.4864, 0). Compute Δw(3) = 0.6 Δw(2) − 0.08 g(3) and the new w."), [r"\Delta w(3)=0.6\times(0.4864,0)-0.08\times(-3.1936,-12)", r"=(0.5473,\ 0.96),\ \ w=(-2.6463,\ 0.36)"], t("Δw(3) = (0.5473, 0.96)；w = (−2.6463, 0.36)。", "Δw(3) = (0.5473, 0.96); w = (−2.6463, 0.36)."), t("核對：上一步 w₂ 沒有動，慣性是 0，梯度那份 0.96 全成了位移。", "Check: w₂ did not move last step, so its inertia is 0 and the whole 0.96 from the gradient becomes displacement."))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 50, 538, q, 18, TXT, True, "left", 84)
        o += chip(x + 16, 150, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 192, 538, 156, "grey", .1) + formula(x + 16, 198, 538, tex[0], 19 if i else 22, 66) + formula(x + 16, 274, 538, tex[1], 19 if i else 20, 66)
        o += chip(x + 16, 362, 538, role, ans, True, 42, 19)
        o += txt(x + 16, 416, 538, chk, 16, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [("p1", page1), ("p2", page2), ("p3", page3), ("p4", page4), ("p5", page5), ("p6", page6), ("p7", page7), ("p8", page8), ("p9", page9), ("p10", page10)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp08_" + fid, fn, W, H, out)
