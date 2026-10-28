"""IDL week 4 KP02 section 1 'darts' page, rebuilt from a picture with image-to-jbd-page.
    python3 build_kp02_darts.py <out_dir>   -> <out_dir>/nodes.json   (then insert_page.py ... --after 3)"""
import math, os, sys
SKILL = os.environ.get("JBD_PAGE_SKILL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "image-to-jbd-page")
sys.path.insert(0, os.path.join(SKILL, "tools"))
from jbd_page import Page

INK, NAVY, RED, GOLD, GREEN, CREAM = "#1c2430", "#1e3053", "#d9544d", "#d49a22", "#2e7d4f", "#f6e4c2"
# numbers of the worked example (true value 10, estimates 7, 9, 8, 10)
X, EST = 10, [7, 9, 8, 10]
mean = sum(EST) / 4; bias = mean - X; var = sum((e - mean) ** 2 for e in EST) / 4; mse = sum((e - X) ** 2 for e in EST) / 4
assert (mean, bias, bias ** 2, var, mse) == (8.5, -1.5, 2.25, 1.25, 3.5) and sum((e - mean) ** 2 for e in EST) == 5.0 and sum((e - X) ** 2 for e in EST) == 14 and var + bias ** 2 == mse


def build():
    P = Page(src_w=1376, src_h=768, prefix="k02d")
    P.rect(0, 0, 1376, 768, fill="#ffffff", z=0)
    P.text(38, 36, 570, 52, "Section 1: 用飛鏢看三種量法", size=40, color=INK, bold=True)
    P.text(600, 38, 760, 50, "(Three ways to measure, seen as darts)", size=36, color=INK)
    P.text(42, 100, 560, 44, "如何估計真實值 X？就像朝著靶心射飛鏢", size=28, color=INK, bold=True)
    P.text(590, 102, 780, 44, "(Estimating a true value X like shooting darts at a target)", size=25, color=INK)
    P.line([(688, 170), (688, 742)], "#222831", 2)
    P.label(40, 176, 620, 44, "飛鏢靶心隱喻", size=31, color=INK)
    P.label(720, 176, 620, 44, "實例演算：四支飛鏢驗證", size=31, color=INK)
    # ---- target
    cx, cy = 231, 432
    for r, col in ((180, NAVY), (147, "#ffffff"), (113, NAVY), (54, "#ffffff")): P.ellipse(cx, cy, r, r, fill=col, z=2)
    P.ellipse(cx, cy, 19, 19, fill="#d90d0d", z=3)
    P.rect(142, 348, 178, 68, fill=NAVY, radius=8, z=4)
    P.label(106, 350, 250, 36, "真實值 X = 10", size=26, color="#ffffff", z=5)
    P.label(106, 382, 250, 30, "(True Value)", size=21, color="#ffffff", bold=False, z=5)
    P.line([(433, 340), (433, 440)], RED, 3, z=4)
    P.label(380, 262, 190, 36, "Bias / 偏差", size=28, color=RED, z=5); P.label(380, 296, 190, 36, "(不準)", size=28, color=RED, z=5)
    P.line([(252, 432), (470, 432)], RED, 5, arrow="both", head=20, z=5)
    P.rect(298, 400, 104, 30, fill=NAVY, radius=6, z=5)
    P.label(296, 400, 108, 30, "距離 = 1.5", size=22, color="#ffffff", z=6)
    # dashed estimates + pale cloud
    P.ellipse(472, 532, 78, 78, fill=CREAM, z=2)
    darts = [(512, 490), (420, 527), (500, 553), (437, 582)]
    for dx, dy in darts: P.line([(cx + 20, cy + 6), (dx, dy)], GOLD, 2, dash=True, z=4)
    for dx, dy in darts:
        P.line([(472, 532), (dx - (dx - 472) * .12, dy - (dy - 532) * .12)], GOLD, 3, arrow="end", head=11, z=4)
        P.ellipse(dx, dy, 9, 9, fill="#16181c", z=5)
        for a in range(0, 360, 60): P.line([(dx, dy), (dx + 15 * math.cos(math.radians(a)), dy + 15 * math.sin(math.radians(a)))], "#16181c", 2, z=5)
    P.ellipse(472, 532, 6, 6, fill=GOLD, z=6)
    P.line([(478, 432), (528, 440), (566, 474), (582, 524)], GOLD, 4, smooth=True, arrow="end", head=16, z=5)
    P.line([(392, 552), (402, 600), (440, 626), (494, 620)], GOLD, 4, smooth=True, arrow="both", head=16, z=5)
    P.label(430, 366, 260, 34, "Variance / 變異數", size=27, color=GOLD, z=5); P.label(430, 400, 260, 34, "(不穩)", size=27, color=GOLD, z=5)
    P.label(440, 584, 250, 40, "估計期望值 E[$\\hat X$] = 8.5", size=22, color=INK, bold=False, z=6)
    P.label(440, 618, 250, 28, "(Expected Estimate)", size=21, color=INK, bold=False, z=6)
    # true value label + brace
    P.line([(cx, 442), (cx, 652)], "#111", 2, dash=True, z=3)
    P.label(30, 632, 220, 36, "真實值 X = 10", size=27, color=INK, z=5); P.label(30, 666, 220, 30, "(True Value)", size=23, color=INK, bold=False, z=5)
    P.line([(232, 650), (232, 662), (436, 662), (436, 676)], "#222", 2.2, z=4); P.line([(436, 662), (640, 662), (640, 650)], "#222", 2.2, z=4)
    P.label(250, 694, 370, 34, "MSE / 均方誤差", size=28, color=INK, z=5); P.label(250, 726, 370, 34, "(總誤差)", size=28, color=INK, z=5)
    # ---- right: given box + four steps
    P.rect(723, 232, 615, 56, fill="#ffffff", stroke="#e6e6e6", sw=1, radius=12, shadowColor="rgba(0,0,0,0.25)", shadowOpacity=1, shadowBlur=8, shadowOffsetX=0, shadowOffsetY=3)
    def M(x, cy, w, s_, size, color=INK, h=56): P.text(x, cy - h / 2, w, h, s_, size=size, color=color, math=True, line=1.0)
    P.text(740, 247, 420, 34, "已知：真實值 X = 10，四支飛鏢估計值", size=21, color=INK, bold=True)
    M(1150, 262, 190, "$\\hat X\\in\\{7,9,8,10\\}$", 22, h=40)
    P.text(738, 308, 150, 40, "1. 期望值", size=27, color=NAVY, bold=True)
    M(888, 330, 400, "$\\mathrm{E}[\\hat X]=\\displaystyle\\frac{7+9+8+10}{4}=8.5$", 21)
    P.text(738, 366, 230, 40, "2. 偏差 Bias", size=27, color=RED, bold=True)
    M(968, 388, 405, "$=8.5-10=-1.5\\Rightarrow\\mathrm{Bias}^2=2.25$", 21, h=40)
    P.text(738, 426, 140, 40, "3. 變異數", size=27, color=GOLD, bold=True)
    M(850, 442, 120, "$\\mathrm{Var}(\\hat X)$", 24, color=GOLD, h=40)
    M(968, 440, 408, "$=\\displaystyle\\frac{(7{-}8.5)^2{+}(9{-}8.5)^2{+}(8{-}8.5)^2{+}(10{-}8.5)^2}{4}$", 17)
    M(924, 492, 330, "$=\\displaystyle\\frac{2.25+0.25+0.25+2.25}{4}$", 19)
    M(924, 530, 300, "$=5.0/4=1.25$", 22, h=36)
    P.text(738, 552, 170, 40, "4. 均方誤差", size=27, color=NAVY, bold=True)
    M(884, 570, 80, "$\\mathrm{MSE}$", 24, color=NAVY, h=40)
    M(968, 568, 408, "$=\\displaystyle\\frac{(7{-}10)^2{+}(9{-}10)^2{+}(8{-}10)^2{+}(10{-}10)^2}{4}$", 17)
    M(924, 611, 300, "$=\\displaystyle\\frac{9+1+4+0}{4}$", 19)
    M(924, 645, 300, "$=14/4=3.5$", 22, h=36)
    # ---- green banner
    P.rect(723, 663, 617, 79, fill=GREEN, radius=14, shadowColor="rgba(0,0,0,0.25)", shadowOpacity=1, shadowBlur=8, shadowOffsetX=0, shadowOffsetY=3)
    P.ellipse(749, 703, 18, 18, fill="#ffffff", z=3); P.line([(740, 703), (747, 711), (760, 695)], GREEN, 4, z=5)
    P.text(784, 672, 300, 36, "數學恆等式驗證", size=26, color="#ffffff", bold=True, z=4)
    P.text(786, 706, 440, 34, "$\\mathrm{Var}+\\mathrm{Bias}^2=1.25+2.25=3.5=\\mathrm{MSE}$", size=19, color="#ffffff", math=True, z=4)
    P.text(1222, 708, 118, 32, "(完美恆等！)", size=19, color="#ffffff", bold=True, z=4)
    return P


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); build().dump(os.path.join(out, "nodes.json"))
