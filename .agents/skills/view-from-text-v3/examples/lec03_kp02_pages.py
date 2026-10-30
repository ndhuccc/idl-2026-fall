import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lec03_kit import *

KP = "KP02"; N = 7
f5 = lambda w: 5 * w * w - 4 * w + 1; d5 = lambda w: 10 * w - 4
assert f5(2) == 13 and d5(2) == 16 and abs(f5(0.4) - 0.2) < 1e-12
W = [2.0]
for _ in range(5): W.append(W[-1] - 0.05 * d5(W[-1]))
assert [round(w, 4) for w in W] == [2, 1.2, 0.8, 0.6, 0.5, 0.45] and [round(f5(w), 4) for w in W] == [13, 3.4, 1, 0.4, 0.25, 0.2125]
FATES = []
for eta in (0.02, 0.10, 0.25):
    r = 1 - 10 * eta; e0 = 1.6; FATES.append((eta, r, r * e0, r * r * e0, 5 * (r * e0) ** 2))
assert [round(v[1], 3) for v in FATES] == [0.8, 0.0, -1.5] and [round(v[2], 3) for v in FATES] == [1.28, 0.0, -2.4]
def run_sq(eta, w0=4.0, n=4):
    w = [w0]
    for _ in range(n - 1): w.append(w[-1] - eta * 2 * w[-1])
    return w
S1 = run_sq(1.0); S11 = run_sq(1.1); S01 = run_sq(0.1)
assert S1 == [4, -4, 4, -4] and round(S11[1], 3) == -4.8 and round(S11[2], 3) == 5.76 and round(S11[1] ** 2, 2) == 23.04 and round(S11[2] ** 2, 4) == 33.1776 and round(S01[1], 3) == 3.2 and round(S01[1] ** 2, 2) == 10.24
L = 10; assert (1 / L, 2 / L) == (0.1, 0.2)

def make(lang):
    d = D(lang, KP, N); T = d.T
    # 1 direction -> step
    g = d.page("KP02 只知道方向，這一步怎麼走", "KP02 Knowing only the direction, how far to step")
    g.beat = 1
    ids = chain(g, 50, 130, 1180, 170, [(BLUE, BLUEL, T("一階近似", "first-order estimate"), r"$\Delta f\approx\nabla f^{\top}\Delta w$"), (ORG, ORGL, T("取 $\\Delta w=-\\eta\\nabla f$", "choose $\\Delta w=-\\eta\\nabla f$"), r"$\Delta f\approx-\eta\|\nabla f\|_2^2$"), (GRN, GRNL, T("平方不是負的", "a square is never negative"), r"$\Delta f\le 0$")], ts=22, bs=30, beat0=1)
    g.beat = 4
    g.rect(190, 340, 900, 110, fill=CARD, stroke=GRAY, sw=3, radius=16, z=3); g.text(200, 340, 880, 110, r"$w^{(k+1)}=w^{(k)}-\eta\,\nabla f\big(w^{(k)}\big)$", size=44, color=INK, align="center", vcenter=True, z=8)
    g.text(190, 454, 900, 40, T("梯度下降", "gradient descent"), size=22, color=MUTED, bold=True, align="center", z=8)
    g.beat = 5
    gold(g, 190, 510, 900, 100, T("學習率 $\\eta$ 只是縮放;真正走的距離是 $\\eta\\|\\nabla f\\|_2$。這只是局部近似,所以要實際檢查。", "$\\eta$ only scales the step: the distance is $\\eta\\|\\nabla f\\|_2$. This is a local estimate, so we must check."), size=22)
    banner(g, T("方向、步長、更新後的損失,三件事都要核對", "check the direction, the step and the loss after the update"))
    d.done(g, "沿負梯度走", "Walk along the negative gradient", {0: ("訓練要對參數最小化損失。每一步要選更新量。位移很小時，損失的變化可以用一階近似估計。", "Training minimizes the loss over the parameters; each step needs an update. For small displacements the change in loss can be estimated to first order."), 1: ("一階近似是梯度轉置乘上位移。", "The first-order estimate is the gradient transposed times the displacement."), 2: ("取位移為負的學習率乘梯度，變化約等於負的學習率乘梯度長度的平方。", "Choosing the displacement as minus the learning rate times the gradient, the change is about minus the learning rate times the squared gradient length."), 3: ("平方不會是負的，所以變化不會是正的：沿負梯度走，損失在近似下下降。", "A square is never negative, so the change is never positive: along the negative gradient the loss falls under the approximation."), 4: ("把它寫成更新式就是梯度下降。", "Written as an update this is gradient descent."), 5: ("學習率只是縮放因子，真正走的距離是學習率乘梯度長度。這只是局部近似，位移變大可能失效，所以要實際檢查。", "The learning rate only scales; the distance is the learning rate times the gradient length. It is a local estimate that can fail for large steps, so we check.")})

    # 2 full step
    g = d.page("完整算一步", "One complete step")
    xr, yr = (-0.2, 2.4), (0, 21)
    ux, uy = axes(g, 50, 120, 700, 480, xr, yr)
    g.beat = 1
    curve(g, ux, uy, f5, BLUE, xr, yr); dot(g, ux(2), uy(13), 12, RED); lab(g, ux(2) - 120, uy(13) - 6, 120, "$w_0=2$", 20, RED, "right")
    g.beat = 2
    g.line([(ux(2), uy(13)), (ux(1.2), uy(3.4))], GRN, sw=5, arrow="end", head=18, z=6); dot(g, ux(1.2), uy(3.4), 12, GRN); lab(g, ux(1.2) + 14, uy(3.4) - 40, 140, "$w_1=1.2$", 20, GRN)
    dot(g, ux(0.4), uy(0.2), 12, PUR); lab(g, ux(0.4) - 20, uy(0.2) - 46, 200, T("最小點 $0.4$", "minimum $0.4$"), 20, PUR)
    g.text(50, 604, 700, 34, T("$f(w)=5w^2-4w+1$", "$f(w)=5w^2-4w+1$"), size=20, color=MUTED, bold=True, align="center", z=8)
    g.beat = 1
    box(g, 790, 126, 440, 150, BLUE, BLUEL, T("起點 $w_0=2$", "start $w_0=2$"), r"$f'(2)=16,\ f(2)=13$", ts=22, bs=26)
    g.beat = 2
    box(g, 790, 296, 440, 160, GRN, GRNL, T("$\\eta=0.05$", "$\\eta=0.05$"), r"$w_1=2-0.05\cdot16=1.2$" + "\n" + r"$f(1.2)=3.4<13$", ts=22, bs=26)
    g.beat = 3
    gold(g, 790, 480, 440, 130, T("只檢查負號不夠:方向、步長與更新後的實際損失都要核對", "checking the sign is not enough: verify direction, step and the real loss"), size=21)
    banner(g, T("梯度是正的，參數往左移，方向對了;損失 $13\\to3.4$", "the gradient is positive so the parameter moves left; loss $13\\to3.4$"))
    d.done(g, "完整算一步", "One complete step", {0: ("用二次損失當主例子。最小點在 0.4，最小損失 0.2。從 w0 等於 2 出發，導數 16，損失 13。", "Use a quadratic loss as the main example. The minimum is at 0.4 with loss 0.2. Starting at w0 equal 2: derivative 16, loss 13."), 1: ("學習率 0.05，新參數 1.2，損失 3.4，確實下降。", "With learning rate 0.05 the new parameter is 1.2 and the loss 3.4, a real decrease."), 2: ("梯度是正的，所以參數往左移，方向對了。", "The gradient is positive, so the parameter moves left: the right direction."), 3: ("只檢查負號並不夠，方向、步長與更新後的實際損失都要核對。", "Checking the sign is not enough; verify direction, step and the actual loss after the update.")})

    # 3 table of steps
    g = d.page("越接近最小點，步子越小", "Steps shrink as we near the minimum")
    g.beat = 1
    rows = [["$w_t$"] + [f"${w:g}$" for w in W], [T("損失", "loss") + " $f(w_t)$"] + [f"${f5(w):g}$" for w in W]]
    colw = [210] + [150] * 6
    table(g, 60, 130, colw, ["$t$"] + [str(i) for i in range(6)], rows, rh=84, fs=26)
    g.beat = 2
    box(g, 60, 420, 540, 180, BLUE, BLUEL, T("化簡後", "simplified"), r"$w_{t+1}=0.5\,w_t+0.2$" + "\n" + T("離最小點 $0.4$ 的距離每步減半", "the distance to $0.4$ halves each step"), ts=22, bs=24)
    g.beat = 3
    gold(g, 640, 420, 590, 180, T("越接近最小點，梯度越小，步子也跟著變小，所以後期下降變慢", "nearer the minimum the gradient is smaller, so steps shrink and late progress slows"), size=24)
    banner(g, T("固定學習率,步長自己會變小", "with a fixed learning rate the steps shrink by themselves"))
    d.done(g, "後期變慢", "Late slowdown", {0: ("繼續用學習率 0.05 更新：參數依序是 2、1.2、0.8、0.6、0.5、0.45，損失 13、3.4、1、0.4、0.25、0.2125。", "Keep updating with 0.05: the parameters are 2, 1.2, 0.8, 0.6, 0.5, 0.45 and the losses 13, 3.4, 1, 0.4, 0.25, 0.2125."), 1: ("化簡後更新式是 0.5 乘 w 加 0.2，離最小點的距離每步減半。", "Simplified the update is 0.5 times w plus 0.2, and the distance to the minimum halves each step."), 2: ("越接近最小點梯度越小，步子也變小，所以後期下降變慢。", "Nearer the minimum the gradient is smaller and so are the steps, so late descent is slow."), 3: ("", "")})

    # 4 three fates
    g = d.page("學習率的三種命運", "The three fates of the learning rate")
    g.beat = 1
    g.text(50, 112, 1180, 40, T("同一個函數,誤差 $e_t=w_t-0.4$ 每步乘上 $1-10\\eta$;起點誤差 $e_0=1.6$", "same function: the error $e_t=w_t-0.4$ is multiplied by $1-10\\eta$ each step; $e_0=1.6$"), size=22, color=MUTED, bold=True, align="center", z=8)
    names = [(BLUE, BLUEL, T("太小:安全但很慢", "too small: safe but slow")), (GRN, GRNL, T("適中:一步到底", "just right: one step")), (RED, REDL, T("太大:越走越遠", "too big: runs away"))]
    for i, ((eta, r, e1, e2, gap), (col, light, nm)) in enumerate(zip(FATES, names)):
        g.beat = i + 1
        x = 50 + i * 400
        box(g, x, 170, 370, 330, col, light, nm, rf"$\eta={eta:g}$" + "\n" + rf"$1-10\eta={r:g}$" + "\n" + rf"$e_1={e1:g},\ e_2={e2:g}$" + "\n" + T("損失差", "loss gap") + rf" $5e_1^2={gap:.4g}$", ts=22, bs=24, th=60)
    g.beat = 4
    gold(g, 190, 530, 900, 80, T("對照:起點損失差 $5e_0^2=12.8$;$\\eta=0.25$ 時損失差變成 $28.8$,比起點還大", "compare the start gap $5e_0^2=12.8$; at $\\eta=0.25$ it becomes $28.8$, bigger than the start"), size=21)
    banner(g, T("太小走不到，適中穩定下降，太大越過谷底甚至越走越遠", "too small: slow; just right: steady; too big: overshoots and may run away"))
    d.done(g, "三種命運", "Three fates", {0: ("同一個函數，誤差每一步乘上固定倍率 1 減 10 倍學習率。起點誤差是 1.6。", "For the same function the error is multiplied each step by 1 minus 10 times the learning rate. The start error is 1.6."), 1: ("學習率 0.02，倍率 0.8，安全但很慢。", "Learning rate 0.02: factor 0.8, safe but slow."), 2: ("學習率 0.10，倍率 0，一步到最小點。", "Learning rate 0.10: factor 0, straight to the minimum in one step."), 3: ("學習率 0.25，倍率負 1.5，每步越過谷底而且越走越遠，這叫拍皮球。", "Learning rate 0.25: factor minus 1.5, overshooting the valley each time and running away, the bouncing ball."), 4: ("起點損失差是 12.8，學習率 0.25 時損失差變成 28.8，比起點還大。", "The start gap is 12.8; at 0.25 it becomes 28.8, larger than the start.")})

    # 5 bouncing ball on w^2
    g = d.page("拍皮球:$f(w)=w^2$", "The bouncing ball: $f(w)=w^2$")
    xr, yr = (-6, 6), (-1, 36)
    ux, uy = axes(g, 50, 120, 700, 480, xr, yr)
    g.beat = 1
    curve(g, ux, uy, lambda w: w * w, GRAY, xr, yr)
    for k, wv in enumerate(S01): dot(g, ux(wv), uy(wv * wv), 9, BLUE)
    box(g, 790, 126, 440, 130, BLUE, BLUEL, T("$\\eta=0.1$:縮小", "$\\eta=0.1$: shrinks"), r"$w_1=3.2,\ f=10.24<16$", ts=22, bs=24)
    g.beat = 2
    for k, wv in enumerate(S1): dot(g, ux(wv), uy(wv * wv), 11, ORG)
    g.line([(ux(4), uy(16)), (ux(-4), uy(16))], ORG, sw=4, arrow="both", head=14, z=6)
    box(g, 790, 276, 440, 150, ORG, ORGL, T("$\\eta=1$:拍皮球", "$\\eta=1$: bouncing"), r"$4,-4,4,-4$" + "\n" + T("損失一直是 $16$", "the loss stays $16$"), ts=22, bs=24)
    g.beat = 3
    for wv in S11: dot(g, ux(max(min(wv, 5.9), -5.9)), uy(min(wv * wv, 35)), 9, RED)
    box(g, 790, 446, 440, 160, RED, REDL, T("$\\eta=1.1$:發散", "$\\eta=1.1$: diverges"), r"$w_1=-4.8,\ f=23.04$" + "\n" + r"$w_2=5.76,\ f=33.18$", ts=22, bs=22)
    banner(g, T("越過谷底不等於損失下降:$\\eta=1$ 時參數兩邊來回，損失不變", "overshooting the valley is not a decrease: at $\\eta=1$ the loss never changes"))
    d.done(g, "拍皮球", "The bouncing ball", {0: ("用更簡單的函數 w 平方，起點 4。學習率 0.1：第一步 3.2，損失 10.24，比 16 小，縮小。", "A simpler function w squared with start 4. Learning rate 0.1: the first step is 3.2, loss 10.24 below 16, shrinking."), 1: ("學習率 1：4、負 4、4、負 4，左右拍皮球，損失一直是 16。", "Learning rate 1: 4, minus 4, 4, minus 4, bouncing between the sides with the loss stuck at 16."), 2: ("學習率 1.1：第一步負 4.8，損失 23.04；第二步 5.76，損失 33.18，發散。", "Learning rate 1.1: the first step is minus 4.8 with loss 23.04; the second 5.76 with loss 33.18: divergence."), 3: ("", "")})

    # 6 L-smooth
    g = d.page("事先算出安全步長:$L$-smooth", "A safe step size in advance: $L$-smoothness")
    g.beat = 1
    box(g, 50, 126, 560, 190, BLUE, BLUEL, T("$L$-smooth", "$L$-smooth"), r"$\|\nabla f(u)-\nabla f(v)\|\le L\|u-v\|$" + "\n" + T("梯度的變化有上限", "the gradient changes at a bounded rate"), ts=24, bs=24)
    g.beat = 2
    box(g, 670, 126, 560, 190, ORG, ORGL, T("下降引理", "descent lemma"), r"$f(w-\eta\nabla f)\le f-\eta\big(1-\tfrac{L\eta}{2}\big)\|\nabla f\|^2$", ts=24, bs=24)
    g.beat = 3
    bx0, bx1 = 70, 1210; by = 400
    u = lambda e: bx0 + e / 0.32 * (bx1 - bx0)
    g.rect(u(0), by, u(0.2) - u(0), 50, fill=GRNL, stroke=GRN, sw=3, radius=8, z=3); g.rect(u(0.2), by, u(0.32) - u(0.2), 50, fill=REDL, stroke=RED, sw=3, radius=8, z=3)
    lab(g, u(0), by + 8, u(0.2) - u(0), T("保證下降:$0<\\eta<2/L$", "guaranteed descent: $0<\\eta<2/L$"), 20, GRN, "center"); lab(g, u(0.2), by + 8, u(0.32) - u(0.2), T("可能發散", "may diverge"), 20, RED, "center")
    for e, c_, t_ in ((0.1, GRN, "$\\eta=0.1=1/L$"), (0.2, ORG, "$2/L=0.2$"), (0.25, RED, "$\\eta=0.25$"), (0.02, BLUE, "$\\eta=0.02$")):
        g.line([(u(e), by - 22), (u(e), by + 62)], c_, sw=4, z=6); lab(g, u(e) - 90, by - 64, 180, t_, 20, c_, "center")
    g.text(70, by + 70, 1140, 40, T("主例子 $f''=10$,所以 $L=10$", "main example: $f''=10$, so $L=10$"), size=22, color=MUTED, bold=True, align="center", z=8)
    g.beat = 4
    gold(g, 190, 520, 900, 90, T("$\\eta=1/L$ 是安全選擇,不是唯一或最大的步長;每步嚴格下降還要求梯度不為零", "$\\eta=1/L$ is a safe choice, not the only or largest step; strict decrease also needs a nonzero gradient"), size=21)
    banner(g, T("三個結果與上一頁完全吻合:$\\eta=0.1$ 一步到底,$0.02$ 緩慢,$0.25$ 發散", "all three match the previous page: $0.1$ one step, $0.02$ slow, $0.25$ diverges"))
    d.done(g, "安全步長", "A safe step size", {0: ("能不能事先算出安全的學習率？只要知道損失彎曲得多厲害。L-smooth 的意思是梯度變化有上限，L 是函數的性質。", "Can we compute a safe learning rate in advance? Yes, if we know how curved the loss is. L-smooth means the gradient changes at a bounded rate; L is a property of the function."), 1: ("由此得到下降引理。括號要大於 0 才保證下降，也就是學習率小於 2 除以 L；取 1 除以 L 時每步至少下降二分之一 L 倍的梯度平方。", "This gives the descent lemma. The bracket must be positive for guaranteed descent, so the learning rate is below 2 over L; with 1 over L each step decreases at least a half over L times the squared gradient."), 2: ("主例子二階導數是 10，所以 L 是 10。學習率 0.1 一步到最小點；0.02 緩慢縮小；0.25 超過 0.2 發散。", "In the main example the second derivative is 10, so L is 10. Learning rate 0.1 reaches the minimum in one step; 0.02 shrinks slowly; 0.25 exceeds 0.2 and diverges."), 3: ("要小心過度推論：1 除以 L 是安全選擇，不是唯一或最大的步長；每步嚴格下降要求梯度不為零；實務上深度網路的 L 很難算出有用的值，所以才改調固定的學習率。", "Beware over-reading: 1 over L is a safe choice, not the only or largest step; strict decrease requires a nonzero gradient; in practice L of a deep network is hard to compute usefully, so we tune a fixed learning rate."), 4: ("", "")})

    # 7 convex vs nonconvex
    g = d.page("凸與非凸:保證的不是同一件事", "Convex and non-convex: different guarantees")
    rows = [[T("能保證什麼", "what is guaranteed"), T("最小值存在時,損失可趨向全域最小", "loss can approach the global minimum"), T("通常至多保證梯度趨近零", "typically at most a gradient approaching zero")], [T("損失會不會上升", "can the loss rise"), T("精確梯度下可單調不增", "can be nonincreasing"), T("同一個下降引理也適用", "the same descent lemma applies")], [T("起點的影響", "effect of the start"), T("影響速度與路徑,最小值相同", "speed and path, same minimum"), T("可能走向不同的盆地或駐點", "different basins or stationary points")]]
    g.beat = 1
    table(g, 60, 120, [250, 440, 440], ["", T("凸、光滑、步長適當", "convex, smooth, suitable step"), T("非凸、光滑、步長適當", "non-convex, smooth, suitable step")], rows, rh=110, fs=22, hfs=24, beats=[1, 2, 3])
    g.beat = 4
    gold(g, 190, 520, 900, 90, T("在非凸損失上看到來回擺動，不能直接判定成功或失敗;下降的趨勢也不是收斂的證明", "back-and-forth on a non-convex loss is neither success nor failure; a downward trend is not proof of convergence"), size=21)
    banner(g, T("深度網路的損失是非凸的:有多個盆地與鞍點", "a deep network's loss is non-convex: many basins and saddle points"))
    d.done(g, "凸與非凸", "Convex vs non-convex", {0: ("凸損失沒有較差的局部最低點。非凸損失可能有多個盆地與鞍點。", "A convex loss has no worse local minima. A non-convex loss can have many basins and saddle points."), 1: ("凸、光滑、步長適當時，損失可趨向全域最小；非凸通常至多保證梯度趨近零。", "For convex, smooth losses with a suitable step the loss can approach the global minimum; non-convex typically guarantees at most a gradient approaching zero."), 2: ("起點在凸時影響速度與路徑，在非凸時可能走向不同的盆地。", "The start affects speed and path when convex, but can lead to different basins when non-convex."), 3: ("在非凸損失上看到來回擺動，不能直接判定成功或失敗，要分清參數的擺動、全資料損失與批次損失。", "Oscillation on a non-convex loss is neither success nor failure; separate parameter oscillation, full-data loss and batch loss.")})
    return d.pages
