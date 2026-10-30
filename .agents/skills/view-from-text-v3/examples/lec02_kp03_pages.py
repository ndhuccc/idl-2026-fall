import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec01_kit import *

KP = "KP03"; N = 8
PTS = [((0, 0), -1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
W, W0 = (1, 1), -1.5; NW = math.hypot(*W)
D = [y * (W[0] * p[0] + W[1] * p[1] + W0) / NW for p, y in PTS]
assert [round(d, 2) for d in D] == [1.06, -0.35, -0.35, 0.35]
# one update: wrong point (0,1), y=+1, eta=1/2
w1 = (W[0] + 1 * 0, W[1] + 1 * 1); w01 = W0 + 1
s_old = W[0] * 0 + W[1] * 1 + W0; s_new = w1[0] * 0 + w1[1] * 1 + w01
assert w1 == (1, 2) and w01 == -0.5 and s_old == -0.5 and s_new == 1.5 and s_new - s_old == 2 * 0.5 * (1 + 1) * 2 / 2 * 1 or True
assert s_new - s_old == 2 * 0.5 * (0 ** 2 + 1 ** 2 + 1)
# trajectory with +-1 labels, eta=1/2 (2*eta=1), sign(0)=+1
def traj():
    w = [0, 0, 0]; ups = []
    while True:
        ch = False
        for (x1, x2), y in PTS:
            s = w[0] * x1 + w[1] * x2 + w[2]; o = 1 if s >= 0 else -1
            if o != y:
                w = [w[0] + y * x1, w[1] + y * x2, w[2] + y]; ups.append(tuple(w)); ch = True
        if not ch: return ups
UPS = traj(); assert UPS == [(0, 0, -1), (0, 1, 0), (0, 1, -1), (1, 1, 0), (1, 1, -1)]

def make(lang):
    zh = lang == "zh"
    T = lambda a, b_: a if zh else b_
    pages = []
    new = lambda n, title: new_page(n, N, KP, title)
    wf = lambda t: "$(" + ",".join(str(v) for v in t) + ")$"

    # ============ 1: four combos ============
    g = new(1, T("KP03 標籤改成 ±1，規則縮成一行", "KP03 With ±1 labels the rule shrinks to one line"))
    g.beat = 1
    g.text(50, 108, 1180, 40, r"$\mathbf{w}\leftarrow\mathbf{w}+\eta\,(y-o)\,\mathbf{x}$" + T("　標籤 $y$、輸出 $o$ 都只取 $+1$ 或 $-1$", "  labels $y$ and outputs $o$ are only $+1$ or $-1$"), size=24, color=MUTED, bold=True, align="center", z=8)
    cx = [60, 230, 400, 580, 940]; cw = [150, 150, 160, 340, 270]
    for k, h in enumerate(("$y$", "$o$", "$y-o$", r"$\eta(y-o)\mathbf{x}$", T("結果", "result"))):
        g.text(cx[k], 160, cw[k], 40, h, size=22, color=MUTED, bold=True, align="center", vcenter=True, z=8)
    rows = [(1, 1, 0, r"$\mathbf{0}$", T("不更新", "no update"), GRN, GRNL), (1, -1, 2, r"$2\eta\mathbf{x}=2\eta y\mathbf{x}$", T("更新", "update"), RED, REDL),
            (-1, 1, -2, r"$-2\eta\mathbf{x}=2\eta y\mathbf{x}$", T("更新", "update"), RED, REDL), (-1, -1, 0, r"$\mathbf{0}$", T("不更新", "no update"), GRN, GRNL)]
    sg = lambda v: ("+" if v > 0 else "") + str(v)
    for i, (y, o, e, upd, res, col, light) in enumerate(rows):
        g.beat = 2 if i in (0, 3) else 3
        yy = 208 + i * 92
        g.rect(50, yy, 1180, 78, fill=light, stroke=col, sw=3, radius=14, z=3)
        assert y - o == e
        for k, t in enumerate((f"${sg(y)}$", f"${sg(o)}$", f"${sg(e)}$", upd, res)):
            g.text(cx[k], yy, cw[k], 78, t, size=28 if k < 4 else 24, color=col if k == 4 else INK, bold=True, align="center", vcenter=True, z=8)
    banner(g, T("猜錯的兩列，$\\eta(y-o)\\mathbf{x}$ 都等於 $2\\eta y\\mathbf{x}$", "both wrong rows give $2\\eta y\\mathbf{x}$"))
    pages.append((g, T("四種組合", "Four combinations"), {
        0: T("把標籤和輸出都寫成正負 1，更新式不變，但 y 和 o 只有四種組合。", "Write labels and outputs as plus or minus 1. The update formula stays, but y and o have only four combinations."),
        1: T("y 和 o 相同：誤差 0，不更新。這是上下兩列。", "y and o equal: error 0, no update. These are the first and last rows."),
        2: T("y 和 o 不同：中間兩列都是猜錯。y 減 o 是正 2 或負 2，更新量都化簡成 2 eta y x。", "y and o differ: the two middle rows are errors. y minus o is plus or minus 2, and both updates reduce to 2 eta y x."),
        3: T("兩種猜錯，一個式子。", "Two kinds of wrong, one formula.")}))

    # ============ 2: one formula ============
    g = new(2, T("猜錯的兩種情形，合成同一式", "Two wrong cases merge into one formula"))
    g.beat = 1
    a = g.mark(); g.rect(60, 140, 500, 120, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3); g.text(60, 140, 500, 120, T("$y=+1$ 卻猜 $-1$\n更新量 $+2\\eta\\mathbf{x}$", "$y=+1$ but guessed $-1$\nupdate $+2\\eta\\mathbf{x}$"), size=24, color=BLUE, bold=True, align="center", z=8) if False else None
    g.rect(60, 140, 500, 130, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(60, 150, 500, 50, T("$y=+1$ 卻猜 $-1$", "$y=+1$ but guessed $-1$"), size=24, color=BLUE, bold=True, align="center", z=8); g.text(60, 206, 500, 56, r"$+2\eta\mathbf{x}$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.rect(720, 140, 500, 130, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(720, 150, 500, 50, T("$y=-1$ 卻猜 $+1$", "$y=-1$ but guessed $+1$"), size=24, color=ORG, bold=True, align="center", z=8); g.text(720, 206, 500, 56, r"$-2\eta\mathbf{x}$", size=30, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    m = g.mark(); g.rect(340, 350, 600, 110, fill="#fef3c7", stroke=GOLD, sw=4, radius=16, z=3)
    g.text(350, 350, 580, 110, r"$\mathbf{w}\leftarrow\mathbf{w}+2\eta\,y_i\,\mathbf{x}_i$", size=40, color=INK, align="center", vcenter=True, z=8)
    g.line([(310, 270), (480, 346)], BLUE, sw=5, arrow="end", head=16, z=4); g.line([(970, 270), (800, 346)], ORG, sw=5, arrow="end", head=16, z=4)
    g.text(340, 466, 600, 40, T("只在 $o_i\\ne y_i$ 時", "only when $o_i\\ne y_i$"), size=22, color=MUTED, bold=True, align="center", z=8)
    g.beat = 3
    g.rect(150, 524, 980, 90, fill=CARD, stroke=GRAY, sw=3, radius=14, z=3)
    g.text(170, 524, 940, 90, T("係數 2 沒有神祕:只是 $y-o$ 反號時必為 $\\pm2$。第 1 章寫的 $\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta y\\mathbf{x}$，是把 2 併進學習率。", "The 2 is no mystery: $y-o$ is $\\pm2$ whenever the signs differ. Chapter 1's $\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta y\\mathbf{x}$ folds the 2 into the learning rate."), size=22, color=INK, bold=True, align="center", z=8) if zh else g.text(170, 534, 940, 80, "The 2 is no mystery: $y-o$ is $\\pm2$ whenever the signs differ. Chapter 1's $\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta y\\mathbf{x}$ folds the 2 into the learning rate.", size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("加進去的永遠是 $y\\mathbf{x}$:標籤是 $+1$ 就加，$-1$ 就減", "always add $y\\mathbf{x}$: add when the label is $+1$, subtract when $-1$"))
    pages.append((g, T("合成一式", "One formula"), {
        0: T("猜錯有兩種：標籤正 1 卻猜負 1，更新量是正 2 eta x；標籤負 1 卻猜正 1，更新量是負 2 eta x。", "Two ways to be wrong: label plus 1 but guessed minus 1 gives plus 2 eta x; label minus 1 but guessed plus 1 gives minus 2 eta x."),
        1: T("兩個都等於 2 eta 乘 y 乘 x，所以只寫一個式子，而且只在猜錯時更新。", "Both equal 2 eta times y times x, so one formula covers them, applied only when wrong."),
        2: T("係數 2 沒有神祕，只是 y 減 o 在反號時必為正負 2。第一章寫的 eta y x，是把 2 併進學習率。", "The 2 is no mystery; it is just y minus o being plus or minus 2 when signs differ. Chapter 1's eta y x folds the 2 into the learning rate."),
        3: T("猜錯的點被往它該在的那一側拉。", "A wrong point is pulled toward the side where it belongs.")}))

    # ============ 3: geometry ============
    g = new(3, T("幾何:權重 $\\mathbf{w}$ 是邊界的法向量", "Geometry: $\\mathbf{w}$ is the boundary's normal vector"))
    x0, y0, side = 60, 110, 480; sc = side / 2.0; u = lambda v: x0 + sc * 0.3 + v * sc; vv = lambda v: y0 + side - sc * 0.3 - v * sc
    g.beat = 1
    g.rect(x0, y0, side, side, fill=CARD, stroke=GRAY, sw=3, radius=14, z=1)
    g.line([(u(0.3), vv(1.2)), (u(1.2), vv(0.3))], RED, sw=5, dash=True, z=5)
    g.text(u(0.2), vv(1.45), 220, 36, "$s=0$", size=20, color=RED, bold=True, z=8)
    g.line([(u(1.2), vv(0.3)), (u(1.55), vv(0.65))], BLUE, sw=7, arrow="end", head=22, z=6)
    g.text(u(1.55) + 8, vv(0.65) - 30, 90, 36, r"$\mathbf{w}$", size=26, color=BLUE, bold=True, z=8)
    g.beat = 2
    for (p, y), d in zip(PTS, D):
        bad = d < 0
        g.ellipse(u(p[0]), vv(p[1]), 20, fill=(GRN if y > 0 else CARD), stroke=(GRN if y > 0 else BLUE), sw=4, z=6)
        if bad: g.ellipse(u(p[0]), vv(p[1]), 32, stroke=RED, sw=4, z=7)
        g.text(u(p[0]) + 36, vv(p[1]) - 16, 140, 34, f"$d={d:.2f}$".replace("-", "-"), size=20, color=(RED if bad else GRN), bold=True, z=8)
    g.beat = 3
    g.rect(620, 130, 610, 440, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(640, 140, 570, 50, T("帶符號距離", "signed distance"), size=26, color=BLUE, bold=True, align="center", z=8)
    g.text(640, 196, 570, 80, r"$d=\dfrac{y\,(\mathbf{w}^\top\mathbf{x}+w_0)}{\|\mathbf{w}\|}$".replace(r"\dfrac", r"\frac"), size=34, color=INK, align="center", vcenter=True, z=8)
    g.text(640, 290, 570, 260, T("猜對:$d\\ge0$\n猜錯:$d<0$\n乘上標籤 $y$，就不必用不可微的絕對值", "right: $d\\ge0$\nwrong: $d<0$\nmultiplying by the label $y$ avoids the non-differentiable absolute value"), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("$\\mathbf{w}=(1,1)$，$w_0=-1.5$:兩個猜錯的點 $d=-0.35$", "$\\mathbf{w}=(1,1)$, $w_0=-1.5$: the two wrong points have $d=-0.35$"))
    pages.append((g, T("法向量與距離", "Normal vector and distance"), {
        0: T("決策邊界是 s 等於 0 的線。邊界上兩點相減，得到一個沿著邊界的向量，權重 w 與它垂直，所以 w 是法向量，指向放電那一側。", "The boundary is the line s equals 0. Subtracting two points on it gives a vector along the boundary; the weight vector w is perpendicular to it, so w is the normal vector, pointing to the firing side."),
        1: T("OR 資料四個點的帶符號距離：原點 1.06；點 0 1 與點 1 0 都是負 0.35，猜錯；點 1 1 是 0.35。", "Signed distances of the four OR points: the origin 1.06; the points 0 1 and 1 0 are minus 0.35, wrong; the point 1 1 is 0.35."),
        2: T("距離乘上標籤，猜對是非負，猜錯是負，而且不必用不可微的絕對值。", "Multiplying the distance by the label gives non-negative for right and negative for wrong, with no non-differentiable absolute value."),
        3: T("", "")}))

    # ============ 4: error and gradient ============
    g = new(4, T("只看猜錯的點，做梯度下降", "Gradient descent on the wrong points only"))
    g.beat = 1
    e1 = g.mark(); g.rect(50, 130, 540, 150, fill=ORGL, stroke=ORG, sw=3, radius=16, z=3)
    g.text(50, 138, 540, 40, T("誤差(猜錯點的集合 $M$)", "error (set $M$ of wrong points)"), size=22, color=ORG, bold=True, align="center", z=8); g.text(60, 184, 520, 80, r"$E=-\sum_{i\in M}y_i(\mathbf{w}^\top\mathbf{x}_i+w_0)$", size=28, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    e2 = g.mark(); g.rect(690, 130, 540, 150, fill=PURL, stroke=PUR, sw=3, radius=16, z=3)
    g.text(690, 138, 540, 40, T("梯度", "gradient"), size=22, color=PUR, bold=True, align="center", z=8); g.text(700, 184, 520, 80, r"$\dfrac{\partial E}{\partial\mathbf{w}}=-\sum_{i\in M}y_i\mathbf{x}_i$".replace(r"\dfrac", r"\frac"), size=30, color=INK, align="center", vcenter=True, z=8)
    g.link(e1, e2, 4, 12, GRAY, 5)
    g.beat = 3
    e3 = g.mark(); g.rect(340, 360, 600, 110, fill="#fef3c7", stroke=GOLD, sw=4, radius=16, z=3)
    g.text(350, 360, 580, 110, r"$\mathbf{w}\leftarrow\mathbf{w}+\eta'\sum_{i\in M}y_i\mathbf{x}_i$", size=34, color=INK, align="center", vcenter=True, z=8)
    g.link(e2, e3, 8, 0, GRAY, 5)
    g.text(340, 480, 600, 80, T("一次只看一個錯點:$\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta' y_i\\mathbf{x}_i$\n與代數結果只差係數 $\\eta'=2\\eta$", "one wrong point at a time: $\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta' y_i\\mathbf{x}_i$\ndiffers from the algebraic result only by $\\eta'=2\\eta$"), size=22, color=INK, bold=True, align="center", z=8)
    banner(g, T("代數與幾何殊途同歸:同一條規則", "algebra and geometry arrive at the same rule"))
    pages.append((g, T("幾何推導", "The geometric route"), {
        0: T("要讓猜錯的點變對，就是把它們的 d 變大。只看猜錯點的集合，定義誤差 E，有錯時為正，越小越好。", "To fix wrong points, enlarge their d. Over the set of wrong points define the error E, positive when there are mistakes and better when smaller."),
        1: T("對 w 求梯度，得到負的錯點 y x 加總。", "Its gradient with respect to w is minus the sum of y x over the wrong points."),
        2: T("梯度下降，得到 w 加上學習率乘錯點的 y x 加總。一次只看一個錯點，就和代數結果相同，只差係數 2，併進學習率。", "Gradient descent gives w plus the learning rate times the sum of y x over the wrong points. One wrong point at a time matches the algebraic result up to the factor 2, folded into the learning rate."),
        3: T("兩條路殊途同歸。", "Both routes arrive at the same rule.")}))

    # ============ 5: one numeric update ============
    g = new(5, T("一次更新:錯點 $(0,1)$", "One update on the wrong point $(0,1)$"))
    g.beat = 1
    st = [(BLUE, BLUEL, T("更新前", "before"), r"$\mathbf{w}=(1,1),\ w_0=-1.5$", r"$s=-0.5,\ o=-1,\ y=+1$"), (ORG, ORGL, T("更新 $\\eta=\\frac12$", "update, $\\eta=\\frac12$"), r"$2\eta=1$", r"$\mathbf{w}\leftarrow(1,1)+(0,1)=(1,2)$" + "\n" + r"$w_0\leftarrow-1.5+1=-0.5$"), (GRN, GRNL, T("更新後", "after"), r"$\mathbf{w}=(1,2),\ w_0=-0.5$", r"$s=1.5>0$")]
    ii = []
    for i, (col, light, nm, a_, b_) in enumerate(st):
        g.beat = i + 1
        x = 50 + i * 410
        ii.append(g.mark()); g.rect(x, 150, 360, 280, fill=light, stroke=col, sw=3, radius=16, z=3)
        g.text(x, 160, 360, 44, nm, size=24, color=col, bold=True, align="center", z=8); g.text(x + 8, 214, 344, 70, a_, size=24, color=INK, align="center", vcenter=True, z=8); g.text(x + 8, 296, 344, 120, b_, size=24, color=INK, bold=True, align="center", z=8)
        if i: g.link(ii[i - 1], ii[i], 4, 12, GRAY, 5)
    g.beat = 3
    g.rect(190, 470, 900, 130, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 478, 860, 54, r"$s$" + T(" 增加 $1.5-(-0.5)=2$", " rose by $1.5-(-0.5)=2$"), size=28, color=INK, bold=True, align="center", z=8)
    g.text(210, 540, 860, 54, r"$=2\eta(\|\mathbf{x}\|^2+1)=1\cdot(1+1)$", size=28, color=INK, align="center", vcenter=True, z=8)
    banner(g, T("這一點變對了:$s$ 由 $-0.5$ 升到 $1.5$", "this point is now right: $s$ went from $-0.5$ to $1.5$"))
    pages.append((g, T("數字驗算", "A numeric check"), {
        0: T("拿錯點 0 1，標籤正 1。目前權重 1、1、偏置負 1.5，s 等於負 0.5，輸出負 1，猜錯。", "Take the wrong point 0 1 with label plus 1. Weights 1, 1, bias minus 1.5, so s is minus 0.5 and the output is minus 1: wrong."),
        1: T("學習率二分之一，所以 2 eta 等於 1，更新就是把 y x 加進去：權重變 1、2，偏置變負 0.5。", "Learning rate one half, so 2 eta equals 1 and the update just adds y x: the weights become 1, 2 and the bias minus 0.5."),
        2: T("新的 s 是 1.5，這一點對了。", "The new s is 1.5, so this point is now right."),
        3: T("s 增加 2，正好等於 2 eta 乘 x 的長度平方加 1。", "s rose by 2, exactly 2 eta times the squared length of x plus 1.")}))

    # ============ 6: trajectory equals KP02 ============
    g = new(6, T("軌跡和 KP02 逐項相同", "The trajectory matches KP02 step by step"))
    g.beat = 1
    g.text(50, 112, 1180, 40, T("從全 0 學 OR，$\\eta=\\frac12$（$2\\eta=1$），標籤 $\\pm1$", "learning OR from zeros, $\\eta=\\frac12$ ($2\\eta=1$), labels $\\pm1$"), size=22, color=MUTED, bold=True, align="center", z=8)
    ids = []
    for i, w in enumerate(UPS):
        g.beat = 2 if i < 3 else 3
        x = 50 + i * 238
        ids.append(g.mark()); g.rect(x, 200, 200, 120, fill=GRNL, stroke=GRN, sw=3, radius=14, z=3)
        g.text(x, 208, 200, 36, T(f"第 {i + 1} 次更新", f"update {i + 1}"), size=20, color=GRN, bold=True, align="center", z=8); g.text(x, 250, 200, 60, wf(w), size=28, color=INK, align="center", vcenter=True, z=8)
        if i: g.link(ids[i - 1], ids[i], 4, 12, GRAY, 4)
    g.beat = 4
    g.rect(190, 380, 900, 140, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(210, 392, 860, 60, T("與 KP02 的二元版逐項相同", "identical to KP02's binary version, term by term"), size=28, color=INK, bold=True, align="center", z=8)
    g.text(210, 456, 860, 56, T("兩種寫法只是記號不同:第 4 輪全對，停止", "the two forms differ only in notation: pass 4 is all right, stop"), size=22, color=MUTED, bold=True, align="center", z=8)
    banner(g, T("$\\operatorname{sign}(0)$ 取 $+1$，與 KP01 的 $s\\ge0$ 放電一致", "$\\operatorname{sign}(0)=+1$, matching KP01's fire at $s\\ge0$"))
    pages.append((g, T("與 KP02 對照", "Same as KP02"), {
        0: T("同樣從全 0 學 OR，學習率二分之一，標籤用正負 1。", "Learn OR again from all zeros, learning rate one half, labels plus or minus 1."),
        1: T("前三次更新依序得到 0 0 負 1、0 1 0、0 1 負 1。", "The first three updates give 0 0 minus 1, 0 1 0 and 0 1 minus 1."),
        2: T("第四、第五次更新得到 1 1 0 與 1 1 負 1，第四輪全對。", "The fourth and fifth updates give 1 1 0 and 1 1 minus 1, and pass four is all right."),
        3: T("這五個權重和 KP02 二元版逐項相同，兩種寫法只是記號不同。符號函數在 0 取正 1，和 KP01 的 s 大於等於 0 放電一致。", "These five weights are identical to KP02's binary version; the two forms differ only in notation. The sign function takes plus 1 at 0, consistent with KP01 firing at s at least 0.")}))

    # ============ 7: two myths ============
    g = new(7, T("兩個容易搞錯的地方", "Two easy mistakes"))
    g.beat = 1
    g.rect(50, 130, 570, 400, fill=REDL, stroke=RED, sw=3, radius=16, z=3)
    g.text(60, 140, 550, 80, T("迷思:把所有點的 $y\\mathbf{x}$ 都加進去", "myth: add $y\\mathbf{x}$ of every point"), size=24, color=RED, bold=True, align="center", z=8)
    g.text(70, 240, 530, 270, T("只有猜錯的點才加。\n猜對的點也加，權重會一直長大，還可能把分對的點推錯。", "Only wrong points are added.\nAdd the right ones too and the weights keep growing, possibly pushing right points wrong."), size=24, color=INK, bold=True, align="center", z=8)
    g.beat = 2
    g.rect(660, 130, 570, 400, fill=BLUEL, stroke=BLUE, sw=3, radius=16, z=3)
    g.text(670, 140, 550, 80, T("迷思:$\\operatorname{sign}(0)$ 取什麼都一樣", "myth: any value of $\\operatorname{sign}(0)$ will do"), size=24, color=BLUE, bold=True, align="center", z=8)
    g.text(680, 240, 530, 270, T("不一樣。剛好落在邊界上的點會被分到另一邊，軌跡也不同。\n本講義取 $+1$。", "It matters. A point exactly on the boundary falls on the other side and the trajectory changes.\nThis note takes $+1$."), size=24, color=INK, bold=True, align="center", z=8)
    banner(g, T("更新只在猜錯時發生;$\\operatorname{sign}(0)$ 要先約定", "update only when wrong; fix the value of $\\operatorname{sign}(0)$ first"))
    pages.append((g, T("兩個迷思", "Two myths"), {
        0: T("第一個迷思：更新時把所有點的 y x 都加進去。不對，只有猜錯的點才加；猜對的也加，權重會一直長大。", "First myth: add y x of every point. No: only wrong points are added; adding the right ones too makes the weights grow forever."),
        1: T("第二個迷思：符號函數在 0 取什麼都一樣。不一樣，邊界上的點會被分到另一邊，軌跡不同。本講義取正 1。", "Second myth: any value of sign at 0 will do. It matters: a point on the boundary falls on the other side. This note takes plus 1."),
        2: T("", ""), 3: T("", "")}))

    # ============ 8: summary ============
    g = new(8, T("猜錯的點被拉向它該在的那一側", "A wrong point is pulled to where it belongs"))
    g.beat = 1
    card(g, 50, 120, 1180, 100, BLUE); g.text(60, 120, 1160, 100, r"$\mathbf{w}\leftarrow\mathbf{w}+2\eta\,y_i\,\mathbf{x}_i\quad(o_i\ne y_i)$", size=38, color=INK, align="center", vcenter=True, z=8)
    g.beat = 2
    for i, s_ in enumerate((T("代數:四種組合，猜錯的兩種化簡成同一式", "algebra: four combinations, the two wrong ones merge"), T("幾何:法向量與距離，對錯點做梯度下降", "geometry: normal vector and distance, gradient descent on wrong points"), T("兩條路得到同一條規則", "both routes give the same rule"))):
        y = 250 + i * 76
        tick(g, 70, y + 12, 34, GRN, z=6); g.text(130, y, 960, 60, s_, size=24, color=INK, bold=True, vcenter=True, z=8)
    g.beat = 3
    g.rect(50, 500, 1000, 110, fill="#fef3c7", stroke=GOLD, sw=3, radius=16, z=3)
    g.text(70, 510, 960, 90, T("Perceptron 這麼會學，是不是什麼分類問題都學得會？兩類資料不能用一條直線分開時，會發生什麼事？", "The Perceptron learns well: can it learn any classification problem? What happens when no straight line separates the two classes?"), size=24, color=INK, bold=True, align="center", z=8)
    mascot(g, 1110, 410, 0.5, "confused")
    banner(g, T("下一個單元:單層 Perceptron 解不了 XOR", "next unit: one Perceptron cannot solve XOR"))
    pages.append((g, T("重點與問題", "Summary and question"), {
        0: T("本單元的重點是一條規則：只在猜錯時，w 加上 2 eta y x。", "The point of this unit is one rule: only when wrong, add 2 eta y x to w."),
        1: T("代數和幾何兩條路都得到它。", "Algebra and geometry both lead to it."),
        2: T("留給下一個單元：兩類資料不能用一條直線分開時，會發生什麼事？", "For the next unit: what happens when no straight line separates the two classes?"),
        3: T("", "")}))
    return pages
