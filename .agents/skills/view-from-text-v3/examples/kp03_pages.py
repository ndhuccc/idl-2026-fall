"""KP03 pages designed in HTML (view-from-text-v3, route A)."""
import os, sys, math
SK = os.environ.get("COURSE_SKILLS") or (os.path.join(os.path.abspath(os.environ["COURSE_REPO"]), ".agents", "skills") if os.environ.get("COURSE_REPO") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__)); HERE = HERE[:-5] if HERE.endswith(("/W_zh", "/W_en")) else HERE
sys.path.insert(0, SK + "/view-from-text-v3/examples"); sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, HERE)
from kp17_pages import Axes, tr, take, legend, fmt, sci, txt, named, card, head, chip, formula, svg, BL, RD, VI, GR, YE, OR, TE, INK, TXT, MUT, hexc, build, native, _sup, rgba, rgb
from kp15_pages import box, cardH
import kp03_nums as K
W, H = 1160, 480
CA, CB = "#4f86bf", "#c4572e"; CAt, CBt = "#7fb0e6", "#e8896a"; LINE, GG, ACT = "#e0c85f", "#b7b0a5", "#f0a35a"
COL = {"000": "#6a9fd6", "001": "#5fb8a8", "010": "#e0c85f", "011": "#a98bd0", "100": "#d97757", "101": "#b7b0a5", "110": "#7dbb8c", "111": "#e07a70"}
def row(x, y, w, h, role, a=.16): return card(x, y, w, h, role, a)
def minus(v): return f"{v:+.1f}".replace("-", "−")
def mn(v): return (f"{v:g}").replace("-", "−")
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
    def dot(s, c, col, r=14, ring=None, hollow=False):
        x, y = s.P(c)
        if hollow: return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#1c1712" stroke="{GG}" stroke-width="3" stroke-dasharray="5 4"/>'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}"/>'
LBL = {"000": (-1, 6, "end"), "100": (1, 6, "start"), "010": (-1, 6, "end"), "110": (1, 6, "start"), "001": (-1, 6, "end"), "101": (1, 6, "start"), "011": (-1, -12, "end"), "111": (1, -12, "start")}
def codelab(cb, c, text=None, col="#f2ede4", gap=24, size=17):
    sx, dy, an = LBL[c]; return cb.lab(c, sx * gap, dy, text or c, col, size, False, an)
def clip(poly, g, keep):
    out = []
    for i in range(len(poly)):
        p, q_ = poly[i], poly[(i + 1) % len(poly)]; gp, gq = g(*p), g(*q_)
        ip = (gp > 0) == bool(keep); iq = (gq > 0) == bool(keep)
        if ip: out.append(p)
        if ip != iq: tt = gp / (gp - gq); out.append((p[0] + tt * (q_[0] - p[0]), p[1] + tt * (q_[1] - p[1])))
    return out
def region(c, lo=-3, hi=3):
    pl = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    for g, ch in zip(K.G, c): pl = clip(pl, g, int(ch))
    return pl
def centroid(pl): return (sum(p[0] for p in pl) / len(pl), sum(p[1] for p in pl) / len(pl))
def polyd(ax, pl, col, op, stroke=None, sw=3, dash=None):
    s = " ".join(f"{ax.X(a):.1f},{ax.Y(b):.1f}" for a, b in pl)
    st = f' stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "") if stroke else ""
    ax.raw(f'<polygon points="{s}" fill="{col}" opacity="{op}"{st}/>')
def lines3(ax, arrows=True, lab=True):
    ax.line([(0, -3), (0, 3)], ACT, 4); ax.line([(-3, 0), (3, 0)], ACT, 4); ax.line([(-3, -2), (2, 3)], ACT, 4)
    if lab:
        ax.text(0.08, 2.88, "g₁ = 0", ACT, "start", 16, True, 4, 6); ax.text(2.95, -0.05, "g₂ = 0", ACT, "end", 16, True, 0, 20); ax.text(-2.95, -2.3, "g₃ = 0", ACT, "start", 16, True, 0, 0)
    if arrows:
        def ar(x1, y1, x2, y2): ax.raw(f'<line x1="{ax.X(x1):.1f}" y1="{ax.Y(y1):.1f}" x2="{ax.X(x2):.1f}" y2="{ax.Y(y2):.1f}" stroke="#f2ede4" stroke-width="3"/><circle cx="{ax.X(x2):.1f}" cy="{ax.Y(y2):.1f}" r="5" fill="#f2ede4"/>')
        ar(0, -2.6, 0.6, -2.6); ar(-2.6, 0, -2.6, 0.6); ar(-1.0, 0.0 + 0.0 + (-1 + 1) - 0, -1.0, 0.0) if False else None; ar(0.6, 1.6, 0.25, 1.95)
def plane_ax(x, y, w, h): return Axes(x, y, w, h, (-3, 3), (-3, 3))
def tint(c): return COL[c]

# ------------------------------------------------ 1: two spaces
def page1(lang):
    t = tr(lang); o = ""; pts = [((1, 1), "①"), ((-1, -2), "②"), ((-2, 2), "③")]
    o += cardH(0, 0, 430, 480, "blue", t("輸入平面：三條線與三個點", "Input plane: three lines and three points"))
    ax = plane_ax(30, 50, 370, 370); lines3(ax, arrows=False, lab=False)
    for (px, py), nm in pts: ax.dot(px, py, "#8a5a2b", 13); ax.text(px, py, nm, "#f2ede4", "middle", 16, True, 0, 6)
    o += ax.html() + txt(16, 430, 398, t("每個點拿到一個編碼", "Each point gets one code"), 17, TXT, True, "center", 28)
    o += cardH(446, 0, 410, 480, "teal", t("編碼立方體：每個點住進一個角", "Code cube: each point lives in one corner"))
    cb = Cube(500, 392, 170); a = cb.edges()
    for c in K.CODES: a += cb.dot(c, "#5a544b", 13, hollow=(c == "101")) + codelab(cb, c)
    for (px, py), nm in pts:
        c = K.code(px, py); x, y = cb.P(c); a += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="#8a5a2b"/><text x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">{nm}</text>'
    o += svg(0, 0, W, H, a) + txt(462, 430, 378, t("三個星號分別在 110、000、011", "The three marks sit at 110, 000, 011"), 17, TXT, True, "center", 28)
    o += cardH(872, 0, 288, 480, "violet", t("三個工具", "Three tools"))
    o += named(884, 46, 266, t("分數：把點代入第 i 條線 gᵢ(x) = wᵢᵀx + bᵢ 得到的數。", "Score: the number from plugging a point into line i, gᵢ(x) = wᵢᵀx + bᵢ."), "term:" + t("分數", "Score"), 16, TXT, 100)
    o += named(884, 154, 266, t("階梯規則：分數大於 0 記 1，其餘記 0（剛好等於 0 也記 0）。", "Step rule: a score above 0 gives 1, anything else 0 (exactly 0 also gives 0)."), "term:" + t("階梯規則", "Step rule"), 16, TXT, 100)
    o += named(884, 262, 266, t("位元：點在線的正側（分數大於 0 的一側）記 1，否則記 0。", "Bit: 1 if the point is on the positive side of a line (score above 0), otherwise 0."), "term:" + t("位元", "Bit"), 16, TXT, 100)
    o += named(884, 370, 266, t("編碼：一個點的三個位元連起來，像它的門牌。", "Code: the three bits of a point written in a row, like its house number."), "term:" + t("編碼", "Code"), 16, TXT, 100)
    return o

# ------------------------------------------------ 2: regions
def page2(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "blue", t("三條線切出七塊地，每塊有自己的編碼", "Three lines cut seven plots, each with its own code"))
    ax = plane_ax(40, 50, 560, 400)
    for c in K.CODES:
        pl = region(c)
        if len(pl) >= 3: polyd(ax, pl, COL[c], .30)
    lines3(ax)
    pos = {"000": (-1.6, -1.2), "001": (-2.2, 1.0), "010": (-0.55, 0.45), "011": (-1.6, 2.0), "100": (1.5, -1.5), "110": (1.9, 1.2), "111": (0.7, 2.2)}
    for c in pos:
        px, py = centroid(region(c)); ax.text(px + (0.25 if c == "010" else 0), py + (0.1 if c == "010" else 0), c, "#f2ede4", "middle", 18, True, 0, 6)
    o += ax.html()
    o += cardH(656, 0, 504, 480, "violet", t("三條線與一個例子", "The three lines and an example"))
    o += card(672, 48, 472, 66, "violet", .22) + formula(672, 52, 472, r"g_1=x_1,\ \ g_2=x_2,\ \ g_3=-x_1+x_2-1", 21, 56)
    o += txt(672, 120, 472, t("係數是教學選擇，不是原圖的方程式；箭頭指向分數為正的一側，線上的點記 0。", "The coefficients are a teaching choice, not the original figure's equations; arrows point to the positive side, and points on a line get 0."), 16, MUT, False, "left", 76)
    o += named(672, 204, 472, t("區域：拿到同一個編碼的所有點，合起來是一塊地。", "Cell: all points that get the same code, together, form one plot."), "term:" + t("區域", "Cell"), 16, TXT, 56)
    o += card(672, 270, 472, 120, "teal", .16) + txt(680, 276, 456, t("例子：點 (1, 1)", "Example: the point (1, 1)"), 17, TXT, True, "left", 28) + formula(680, 304, 456, r"g_1=1,\ g_2=1,\ g_3=-1+1-1=-1", 19, 44) + txt(680, 352, 456, t("位元 1, 1, 0（−1 ≤ 0 記 0），編碼是 110。", "Bits 1, 1, 0 (−1 ≤ 0 gives 0), so the code is 110."), 17, TXT, False, "left", 30)
    o += card(672, 402, 472, 58, "yellow", .14) + txt(680, 410, 456, t("七種編碼出現，101 不出現。〔程式計算〕", "Seven codes occur; 101 never does. (computed)"), 18, TXT, True, "center", 42)
    return o

# ------------------------------------------------ 3: no 101
def page3(lang):
    t = tr(lang); o = ""
    conds = [(t("① 第 1 位是 1：x₁ > 0", "① bit 1 is 1: x₁ > 0"), lambda x, y: x > 0, "blue"), (t("② 第 2 位是 0：x₂ ≤ 0", "② bit 2 is 0: x₂ ≤ 0"), lambda x, y: y <= 0, "orange"), (t("③ 第 3 位是 1：x₂ > x₁ + 1", "③ bit 3 is 1: x₂ > x₁ + 1"), lambda x, y: y > x + 1, "teal")]
    sh = [("#6a9fd6", [(0, -3), (3, -3), (3, 3), (0, 3)]), ("#d97757", [(-3, -3), (3, -3), (3, 0), (-3, 0)]), ("#5fb8a8", [(-3, -2), (2, 3), (-3, 3)])]
    for i in range(4):
        x0 = i * 292; o += card(x0, 0, 276, 300, "blue" if i < 3 else "red", .12) + head(x0, 0, 276, ["blue", "orange", "teal", "red"][i], [t("條件 ①", "Condition ①"), t("條件 ②", "Condition ②"), t("條件 ③", "Condition ③"), t("三個疊起來", "All three stacked")][i], 34, 17)
        ax = Axes(x0 + 20, 48, 236, 190, (-3, 3), (-3, 3))
        if i < 3: polyd(ax, sh[i][1], sh[i][0], .38)
        else:
            for col, pl in sh: polyd(ax, pl, col, .2)
        ax.line([(0, -3), (0, 3)], GG, 2); ax.line([(-3, 0), (3, 0)], GG, 2); ax.line([(-3, -2), (2, 3)], GG, 2)
        if i == 3: ax.text(0, 0, t("沒有共同部分", "no common part"), "#f08a80", "middle", 18, True, 0, -60)
        o += ax.html()
        o += txt(x0 + 8, 246, 260, [t("x₁ > 0：右半邊", "x₁ > 0: the right half"), t("x₂ ≤ 0：下半邊", "x₂ ≤ 0: the lower half"), t("線 g₃ 的上方", "above the line g₃"), t("交集是空的", "the intersection is empty")][i], 16, TXT, False, "center", 26)
    o += row(0, 316, 570, 152, "grey", .12) + txt(10, 322, 550, t("為什麼矛盾", "Why they contradict"), 17, MUT, True, "left", 26) + txt(10, 352, 550, t("①② 讓 x₁ > 0、x₂ ≤ 0，所以 x₂ ≤ 0 < x₁ + 1，③ 不可能成立。三位元能寫 8 種編碼，圖上只有 7 塊。", "① and ② give x₁ > 0 and x₂ ≤ 0, so x₂ ≤ 0 < x₁ + 1 and ③ cannot hold. Three bits can write 8 codes, but the plane has only 7 plots."), 17, TXT, False, "left", 108)
    o += row(586, 316, 574, 152, "red", .14) + named(596, 322, 556, t("實際碼：至少有一個輸入點會得到的編碼（7 個）。", "Realized code: a code that at least one input point receives (7 of them)."), "term:" + t("實際碼", "Realized code"), 16, TXT, 56) + named(596, 384, 556, t("虛擬碼：寫得出來，卻沒有任何輸入點得到的編碼（101）；不是資料還沒蒐集到。", "Unrealized code: a code that can be written but no input point receives (101); it is not missing data."), "term:" + t("虛擬碼", "Unrealized code"), 16, TXT, 80)
    return o

# ------------------------------------------------ 4: cube
def page4(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 600, 480, "teal", t("8 個編碼是立方體 8 個角", "The 8 codes are the cube's 8 corners"))
    cb = Cube(180, 400, 190); a = cb.edges()
    for c in K.CODES: a += cb.dot(c, COL[c], 18, hollow=(c == "101")) + codelab(cb, c, None, "#f2ede4", 30, 18)
    o += svg(0, 0, W, H, a) + txt(16, 422, 568, t("角的顏色與上一頁平面上各塊地相同；101 是空心圓，沒有輸入會到達。", "Corner colors match the plots of the previous page; 101 is hollow: no input reaches it."), 17, TXT, False, "center", 52)
    o += cardH(616, 0, 544, 480, "violet", t("最後一層是切一刀", "The last layer is one cut"))
    o += txt(632, 46, 512, t("三位元的 8 種編碼，正好是立方體 8 個角的座標 (y₁, y₂, y₃)；階梯規則只輸出角，不填滿內部。", "The 8 three-bit codes are exactly the coordinates (y₁, y₂, y₃) of the cube's corners; the step rule outputs only corners and never fills the inside."), 16, TXT, False, "left", 76)
    o += named(632, 130, 512, t("輸出分數：q(y) = a₁y₁ + a₂y₂ + a₃y₃ + a₀；q > 0 判 A，否則判 B。", "Output score: q(y) = a₁y₁ + a₂y₂ + a₃y₃ + a₀; q > 0 means A, otherwise B."), "term:" + t("輸出分數", "Output score"), 16, TXT, 56)
    o += card(632, 192, 512, 56, "violet", .22) + formula(632, 198, 512, r"q(\mathbf{y})=a_1y_1+a_2y_2+a_3y_3+a_0", 22, 44)
    o += txt(632, 258, 512, t("aᵢ 是每一位的票數，a₀ 是起始分。", "aᵢ is the vote of each bit and a₀ is the starting score."), 16, MUT, False, "left", 28)
    o += row(632, 296, 512, 76, "teal", .16) + txt(640, 302, 496, t("q = 0 是立方體裡的一個平面：把角分成兩邊，這就是切一刀。", "q = 0 is a plane in the cube: it splits the corners into two sides, which is one cut."), 17, TXT, True, "center", 60)
    o += row(632, 384, 512, 76, "yellow", .14) + txt(640, 390, 496, t("條件只需對實際碼成立；虛擬碼的標籤不會改變任何輸入的預測。", "The condition only has to hold for realized codes; a label on the unrealized code changes no input's prediction."), 17, TXT, False, "center", 60)
    return o

# ------------------------------------------------ 5: score every corner
def cutpoly(cb, f):
    pts = []
    for c in K.CODES:
        for i in range(3):
            if c[i] == "0":
                d = c[:i] + "1" + c[i + 1:]; fc, fd = f(c), f(d)
                if (fc > 0) != (fd > 0):
                    tt = fc / (fc - fd); (x1, y1), (x2, y2) = cb.P(c), cb.P(d); pts.append((x1 + tt * (x2 - x1), y1 + tt * (y2 - y1)))
    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
def page5(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 560, 480, "blue", t("切面 q = 0 與各角分數", "The cut q = 0 and the corner scores"))
    cb = Cube(150, 390, 160); a = ""
    pl = cutpoly(cb, K.q1); a += f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pl)}" fill="{LINE}" opacity=".22" stroke="{LINE}" stroke-width="3"/>' + cb.edges()
    for c in K.CODES:
        ghost = c == "101"; x, y = cb.P(c)
        if ghost: a += cb.dot(c, "", 15, hollow=True)
        elif K.cls1(c) == "A": a += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{CA}"/>'
        else: a += f'<rect x="{x-15:.1f}" y="{y-15:.1f}" width="30" height="30" rx="5" fill="{CB}"/>'
        a += codelab(cb, c, f"{c}: {minus(K.q1(c))}", "#f2ede4", 28, 16)
    o += svg(0, 0, W, H, a) + txt(16, 424, 528, t("黃色是切面 q = 0：A（圓點）在一側，B（方點）在另一側。", "Yellow is the cut q = 0: A (circles) on one side, B (squares) on the other."), 17, TXT, False, "center", 52)
    o += cardH(576, 0, 584, 480, "violet", t("逐角計分：q = −y₁ − y₂ + y₃ + 0.5", "Score every corner: q = −y₁ − y₂ + y₃ + 0.5"))
    for xx, lab, w_ in ((586, t("角 y", "code y"), 70), (660, t("代入", "substitute"), 230), (900, "q", 80), (986, t("類別", "class"), 70)): o += txt(xx, 44, w_, lab, 16, MUT, True, "center", 24)
    for i, c in enumerate([c for c in K.CODES if c != "101"]):
        y0 = 70 + i * 36; A = K.cls1(c) == "A"; o += row(588, y0, 556, 32, "blue" if A else "orange", .2)
        sub = f"{-int(c[0])}{-int(c[1]):+d}{int(c[2]):+d}+0.5".replace("-", "−") if False else f"−{c[0]}−{c[1]}+{c[2]}+0.5"
        o += txt(586, y0 + 3, 70, c, 17, TXT, True, "center", 26) + txt(660, y0 + 3, 230, sub, 17, TXT, False, "center", 26) + txt(900, y0 + 3, 80, minus(K.q1(c)), 17, TXT, True, "center", 26) + txt(986, y0 + 3, 70, K.cls1(c), 17, TXT, True, "center", 26)
    o += txt(588, 326, 556, t("101 的分數是 −1−0+1+0.5 = +0.5，但沒有輸入會得到它，不影響分類。", "The score of 101 is −1−0+1+0.5 = +0.5, but no input gets it, so classification is unaffected."), 16, MUT, False, "left", 52)
    o += named(588, 384, 556, t("線性可分：A 全在平面一側、B 全在另一側。", "Linearly separable: all of A on one side of a plane and all of B on the other."), "term:" + t("線性可分", "Linearly separable"), 16, TXT, 30)
    o += card(588, 420, 556, 44, "green", .2) + txt(596, 426, 540, t("這一刀切得開。〔程式計算〕", "This cut separates them. (computed)"), 18, TXT, True, "center", 30)
    return o

# ------------------------------------------------ 6: union with a gap
def page6(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 640, 480, "blue", t("輸入平面：A 區由三塊地併成，中間有缺口", "Input plane: region A is three plots joined, with a notch"))
    ax = plane_ax(40, 50, 560, 400)
    for c in K.CODES:
        pl = region(c)
        if len(pl) >= 3 and c != "101": polyd(ax, pl, "#6a9fd6" if K.cls1(c) == "A" else "#d97757", .28)
    lines3(ax, arrows=False, lab=False)
    ax.line([K.P, K.Q], "#f2ede4", 3, "9 7")
    for nm, p, col in (("P", K.P, CA), ("Q", K.Q, CA)): ax.dot(p[0], p[1], col, 13); ax.text(p[0], p[1], nm, "#f2ede4", "middle", 16, True, 0, 6)
    ax.rect(K.R[0] - .15, K.R[0] + .15, K.R[1] - .15, K.R[1] + .15, CB, 1); ax.text(K.R[0], K.R[1], "R", "#f2ede4", "middle", 16, True, 0, 6)
    ax.text(0.9, 1.9, t("A 區", "region A"), CAt, "start", 18, True, 0, 0) if False else None
    ax.text(-1.7, -1.0, "A", CAt, "middle", 22, True); ax.text(-2.2, 1.9, "A", CAt, "middle", 22, True); ax.text(1.8, 0.8, "B", CBt, "middle", 22, True); ax.text(0.9, -1.8, "B", CBt, "middle", 22, True)
    o += ax.html()
    o += cardH(656, 0, 504, 480, "violet", t("聯集與凸", "Union and convex"))
    o += named(672, 46, 472, t("聯集：把幾塊區域併成一塊。A 類是 000、001、011 三塊地的聯集。", "Union: several regions joined into one. Class A is the union of the plots 000, 001, 011."), "term:" + t("聯集", "Union"), 16, TXT, 76)
    o += named(672, 130, 472, t("凸：區域內任取兩點，連線都不跑出區域。", "Convex: for any two points of the region, the segment between them stays inside."), "term:" + t("凸", "Convex"), 16, TXT, 56)
    o += card(672, 196, 472, 126, "red", .16) + txt(680, 202, 456, t("取 P(−0.15, −1) 與 Q(−0.15, 2)：兩點都在 A 區，連線 PQ 卻經過 B 塊的 R(−0.15, 0.62)。", "Take P(−0.15, −1) and Q(−0.15, 2): both lie in A, yet the segment PQ passes through R(−0.15, 0.62) of the B plot."), 17, TXT, False, "left", 100)
    o += row(672, 334, 472, 50, "yellow", .16) + txt(680, 342, 456, t("A 區不是凸的。〔程式計算〕", "Region A is not convex. (computed)"), 18, TXT, True, "center", 34)
    o += card(672, 396, 472, 70, "orange", .14) + txt(680, 400, 456, t("迷思：兩層網路只會分類凸區域。單一編碼的區域是凸的，但最後一層可接受多個編碼，其聯集可以有缺口。", "Myth: two-layer networks only classify convex regions. One code's region is convex, but the last layer can accept several codes, whose union may have notches."), 16, TXT, False, "center", 64)
    return o

# ------------------------------------------------ 7: second labelling
def page7(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 430, 480, "blue", t("第二組標籤：A = {000, 110, 111}", "Second labelling: A = {000, 110, 111}"))
    ax = plane_ax(30, 50, 370, 370)
    for c in K.CODES:
        pl = region(c)
        if len(pl) >= 3 and c != "101": polyd(ax, pl, "#6a9fd6" if K.cls2(c) == "A" else "#d97757", .28)
    lines3(ax, arrows=False, lab=False)
    o += ax.html() + txt(16, 430, 398, t("藍色是 A 的三塊地，橘色是其餘四塊。", "Blue: A's three plots; orange: the rest."), 17, TXT, False, "center", 30)
    o += cardH(446, 0, 410, 480, "teal", t("立方體底面 y₃ = 0 的四個角", "The four corners of the bottom face y₃ = 0"))
    cb = Cube(520, 380, 180); a = ""
    pts = " ".join(f"{cb.P(c)[0]:.1f},{cb.P(c)[1]:.1f}" for c in ("000", "100", "110", "010"))
    a += f'<polygon points="{pts}" fill="#e07a70" opacity=".16" stroke="#e07a70" stroke-width="3" stroke-dasharray="9 7"/>' + cb.edges()
    for c in K.CODES:
        x, y = cb.P(c)
        if c == "101": a += cb.dot(c, "", 15, hollow=True)
        elif K.cls2(c) == "A": a += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{CA}"/><text x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">A</text>'
        else: a += f'<rect x="{x-16:.1f}" y="{y-16:.1f}" width="32" height="32" rx="5" fill="{CB}"/><text x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">B</text>'
        a += codelab(cb, c, None, "#f2ede4", 28, 16)
    o += svg(0, 0, W, H, a) + txt(462, 430, 378, t("紅色虛線：底面，A 在 000、110，B 在 010、100", "Red dashes: the bottom face, A at 000, 110 and B at 010, 100"), 16, "#f08a80", True, "center", 48)
    o += cardH(872, 0, 288, 480, "red", t("XOR 又出現了", "XOR returns"))
    for k_, (c, ca) in enumerate((("010", 0), ("110", 1), ("000", 1), ("100", 0))): pass
    sq = [("010", 0, 0), ("110", 1, 0), ("000", 0, 1), ("100", 1, 1)]
    for c, cx, cy in sq:
        x = 930 + cx * 110; y = 70 + cy * 110; A = K.cls2(c) == "A"; o += row(x, y, 100, 100, "blue" if A else "orange", .35) + txt(x, y + 22, 100, K.cls2(c), 24, TXT, True, "center", 32) + txt(x, y + 62, 100, c, 17, TXT, False, "center", 26)
    o += named(884, 296, 266, t("不可分：沒有任何一刀能把 A 與 B 分開；這一組標籤是這組線的一個反例。", "Nonseparable: no cut separates A from B; this labelling is a counterexample for this set of lines."), "term:" + t("不可分", "Nonseparable"), 16, TXT, 100) 
    o += txt(884, 400, 266, t("和 KP01 的 XOR 一樣：沿對角交錯。", "Like XOR in KP01: alternating along the diagonal."), 16, MUT, False, "center", 52)
    return o

# ------------------------------------------------ 8: midpoint proof
def page8(lang):
    t = tr(lang); o = ""
    o += cardH(0, 0, 520, 480, "blue", t("兩條對角線共用一個中點 M", "Two diagonals share one midpoint M"))
    ax = Axes(110, 70, 300, 300, (-0.2, 1.2), (-0.2, 1.2)); ax.line([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], GG, 3)
    ax.line([(0, 0), (1, 1)], CA, 5); ax.line([(0, 1), (1, 0)], CB, 5)
    for p, nm, col in (((0, 0), "000", CA), ((1, 1), "110", CA), ((0, 1), "010", CB), ((1, 0), "100", CB)):
        ax.dot(p[0], p[1], col, 15); ax.text(p[0], p[1], "A" if col == CA else "B", "#f2ede4", "middle", 16, True, 0, 6)
    ax.text(0, 0, "000", "#f2ede4", "end", 16, False, -24, 6); ax.text(1, 1, "110", "#f2ede4", "start", 16, False, 24, 6); ax.text(0, 1, "010", "#f2ede4", "end", 16, False, -24, 6); ax.text(1, 0, "100", "#f2ede4", "start", 16, False, 24, 6)
    ax.dot(.5, .5, "#f2ede4", 9); ax.text(.5, .5, "M", "#f2ede4", "start", 18, True, 14, -12)
    o += ax.html() + txt(16, 392, 488, t("藍線：A 的對角線 000—110；橘線：B 的對角線 010—100。M = (½, ½, 0)", "Blue: A's diagonal 000—110. Orange: B's diagonal 010—100. M = (½, ½, 0)"), 17, TXT, False, "center", 52)
    o += txt(16, 440, 488, t("底面 y₃ = 0（俯視）", "Bottom face y₃ = 0 (top view)"), 16, MUT, False, "center", 24)
    o += cardH(536, 0, 624, 480, "violet", t("證明：任何 a₀, a₁, a₂ 都切不開", "Proof: no a₀, a₁, a₂ separates them"))
    steps = [(r"q(000)+q(110)=a_0+(a_1+a_2+a_0)=a_1+a_2+2a_0>0", t("1. 兩個 A 角相加，必須 > 0", "1. Add the two A corners: must be > 0"), "blue"), (r"q(010)+q(100)=(a_2+a_0)+(a_1+a_0)=a_1+a_2+2a_0\leq 0", t("2. 兩個 B 角相加，必須 ≤ 0", "2. Add the two B corners: must be ≤ 0"), "orange")]
    for i, (tx, lb, role) in enumerate(steps):
        y0 = 48 + i * 92; o += card(552, y0, 592, 84, role, .16) + txt(560, y0 + 4, 576, lb, 16, TXT, True, "left", 24) + formula(560, y0 + 30, 576, tx, 16.5, 46)
    o += row(552, 236, 592, 56, "red", .2) + txt(560, 244, 576, t("3. 同一個式子既要 > 0 又要 ≤ 0：矛盾。〔程式計算〕", "3. One expression cannot be both > 0 and ≤ 0: contradiction. (computed)"), 18, TXT, True, "center", 40)
    o += card(552, 304, 592, 150, "yellow", .14) + txt(560, 310, 576, t("幾何意思", "The geometric meaning"), 17, MUT, True, "left", 26) + formula(560, 340, 576, r"q(M)=\frac{a_1+a_2+2a_0}{2}", 22, 56) + txt(560, 398, 576, t("q 是線性的：q(M) 既是兩個 A 分數的平均（正），也是兩個 B 分數的平均（非正）。", "q is linear: q(M) is both the average of the two A scores (positive) and of the two B scores (non-positive)."), 16, TXT, False, "left", 52)
    return o

# ------------------------------------------------ 9: slice corners
def page9(lang):
    t = tr(lang); o = ""
    tg = [("110", r"y_1+y_2-y_3=1.5"), ("111", r"y_1+y_2+y_3=2.5"), ("000", r"-y_1-y_2-y_3=-0.5")]
    o += card(0, 0, 1160, 66, "blue", .14) + txt(16, 10, 1128, t("再加一層：每個目標角用一個平面單獨切下來，再把結果併起來", "One more layer: cut each target corner off with its own plane, then merge the results"), 20, TXT, True, "center", 46)
    for i, (c, eq) in enumerate(tg):
        x0 = i * 392; o += card(x0, 76, 376, 322, "green", .1) + head(x0, 76, 376, "green", t(f"切下角 {c}", f"Cut off corner {c}"), 34, 17)
        cb = Cube(x0 + 110, 330, 120); a = cb.edges()
        for cc in K.CODES:
            x, y = cb.P(cc)
            if cc == c: a += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="#4f9a68"/><text x="{x:.1f}" y="{y+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">★</text>'
            else: a += cb.dot(cc, "#5a544b", 8, hollow=(cc == "101" and False))
        o += svg(0, 0, W, H, a)
        o += txt(x0 + 8, 360, 360, t(f"平面 {eq.replace('=','＝') if False else ''}", ""), 16) if False else ""
        o += formula(x0 + 8, 346, 360, eq, 19, 40)
    o += card(0, 408, 1160, 64, "yellow", .14) + txt(16, 414, 1128, t("三個平面合起來認出整個 A 集合：這就是再加一層。被擋住的是「這組固定的線加上這一刀」，不是所有兩層網路。", "Together the three planes recognise the whole A set: that is one more layer. What is blocked is this fixed set of lines plus one cut, not every two-layer network."), 17, TXT, True, "center", 52)
    return o

# ------------------------------------------------ 10: hypercubes
def page10(lang):
    t = tr(lang); o = ""; a = ""
    o += cardH(0, 0, 760, 480, "blue", t("每多一個位元，角數加倍", "Each extra bit doubles the number of corners"))
    # segment
    o += txt(30, 60, 160, t("1 位元", "1 bit"), 17, TXT, True, "center", 26) + txt(30, 86, 160, t("2 個角", "2 corners"), 16, MUT, False, "center", 24)
    a += f'<line x1="60" y1="230" x2="160" y2="230" stroke="{GG}" stroke-width="3"/>' + "".join(f'<circle cx="{x}" cy="230" r="12" fill="{CA}"/>' for x in (60, 160))
    o += txt(250, 60, 190, t("2 位元", "2 bits"), 17, TXT, True, "center", 26) + txt(250, 86, 190, t("4 個角", "4 corners"), 16, MUT, False, "center", 24)
    a += f'<rect x="290" y="180" width="100" height="100" fill="none" stroke="{GG}" stroke-width="3"/>' + "".join(f'<circle cx="{x}" cy="{y}" r="12" fill="{CA}"/>' for x in (290, 390) for y in (180, 280))
    o += txt(500, 60, 230, t("3 位元", "3 bits"), 17, TXT, True, "center", 26) + txt(500, 86, 230, t("8 個角", "8 corners"), 16, MUT, False, "center", 24)
    cb = Cube(530, 300, 100); a += cb.edges() + "".join(cb.dot(c, CA, 11) for c in K.CODES)
    o += svg(0, 0, W, H, a)
    o += txt(16, 340, 728, t("2 → 4 → 8 → … → 2ᵖ", "2 → 4 → 8 → … → 2ᵖ"), 24, TXT, True, "center", 34)
    o += txt(16, 390, 728, t("每一條線給一位元，p 條線的編碼住在 p 維立方體的 2ᵖ 個角。〔程式計算〕", "Each line gives one bit, so codes of p lines live in the 2ᵖ corners of a p-dimensional cube. (computed)"), 17, TXT, False, "center", 56)
    o += cardH(776, 0, 384, 480, "violet", t("一般寫法", "The general form"))
    o += txt(792, 46, 352, t("輸入有 d 個座標，有 p 條線：", "The input has d coordinates and there are p lines:"), 16, TXT, False, "left", 52)
    o += card(792, 100, 352, 96, "violet", .22) + formula(792, 108, 352, r"\mathbf{y}=[f(g_1(\mathbf{x})),\dots,f(g_p(\mathbf{x}))]^T\in\{0,1\}^p", 17, 80)
    o += card(792, 208, 352, 80, "teal", .16) + formula(792, 214, 352, r"f(t)=1\ (t>0),\quad f(t)=0\ (t\leq 0)", 18, 46)
    o += named(792, 304, 352, t("超立方體：p 維的立方體；{0,1}ᵖ 的 2ᵖ 個點是它的角。", "Hypercube: a p-dimensional cube; the 2ᵖ points of {0,1}ᵖ are its corners."), "term:" + t("超立方體", "Hypercube"), 16, TXT, 80)
    o += row(792, 398, 352, 64, "yellow", .14) + txt(800, 404, 336, t("本講義的例子 d = 2、p = 3。", "This unit's example has d = 2 and p = 3."), 17, TXT, True, "center", 50)
    return o

# ------------------------------------------------ 11: exercises
def page11(lang):
    t = tr(lang); o = ""
    qs = [("blue", t("題 1", "Q1"), t("把虛擬碼 101 指定為 A 或 B，輸入上的預測會變嗎？", "If the unrealized code 101 is assigned A or B, do the predictions on inputs change?"), [r"g_1=x_1>0,\ g_2=x_2\leq 0", r"g_3=-x_1+x_2-1>0\Rightarrow x_2>x_1+1>1", r"(x_2>1)\ \wedge\ (x_2\leq 0)\ \Rightarrow\ \times"], t("不會：沒有任何輸入得到 101。", "No: no input gets 101."), t("核對：101 只影響不存在的輸入，所有輸入的預測不變。", "Check: 101 affects only non-existent inputs, so every prediction stays.")),
          ("red", t("題 2", "Q2"), t("第二組標籤中，只改 a₃，能消除底面四點的矛盾嗎？", "In the second labelling, can changing only a₃ remove the contradiction on the bottom face?"), [r"S=q(000)+q(110)=q(010)+q(100)", r"S=a_1+a_2+2a_0,\ a_3\ \mathrm{absent}", r"a_1=a_2=0,\ a_0=1:\ q=1"], t("不能：a₃ 消失，矛盾依舊。", "No: a₃ drops out and the contradiction stays."), t("核對：B 被判成 A，與 a₃ 無關。〔程式計算〕", "Check: B is called A whatever a₃ is. (computed)"))]
    for i, (role, name, q, tex, ans, chk) in enumerate(qs):
        x = i * 590; o += card(x, 0, 570, 470, role) + head(x, 0, 570, role, name, 36, 18)
        o += txt(x + 16, 50, 538, q, 19, TXT, True, "left", 84)
        o += chip(x + 16, 146, 130, "grey", t("解題示範", "Solution"), False, 30, 16)
        o += card(x + 16, 190, 538, 150, "grey", .1)
        for j, tx in enumerate(tex): o += formula(x + 16, 196 + j * 48, 538, tx, 18, 46)
        o += chip(x + 16, 356, 538, role, ans, True, 42, 19)
        o += txt(x + 16, 410, 538, chk, 16, MUT, False, "left", 48)
    return o

PAGES_DESIGN = [(f"p{i}", globals()[f"page{i}"]) for i in range(1, 12)]
if __name__ == "__main__":
    out = sys.argv[1]; only = sys.argv[2:]
    for fid, fn in PAGES_DESIGN:
        if only and fid not in only: continue
        print(fid); build("kp03_" + fid, fn, W, H, out)
