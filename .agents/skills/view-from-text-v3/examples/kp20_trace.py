"""KP20 page 5 (hand trace of the kernel perceptron on four XOR-type points), HTML design -> native nodes (view-from-text-v3)."""
import os, sys
import numpy as np
SK = "/home/user/pr-2026-fall/.agents/skills"
sys.path.insert(0, SK + "/view-from-text-v3/tools")
from htmlkit import *
formula = k
W, H = 1160, 470
BL, RD, GR, OR, YE = (hexc(c) for c in ("blue", "red", "green", "orange", "yellow"))
def tr(lang): return lambda zh, en: zh if lang == "zh" else en
def trace():
    P = np.array([[1, 0], [-1, 0], [0, 1], [0, -1.]]); y = np.array([1, 1, -1, -1.]); G = (P @ P.T + 1) ** 2; a = np.zeros(4); rounds = []
    for ep in range(3):
        rows = []
        for i in range(4):
            g = (a * y) @ G[:, i] + (a * y).sum(); bad = y[i] * g <= 0
            if bad: a[i] += 1
            rows.append((i + 1, int(round(g)), int(round(y[i] * g)), bool(bad)))
        rounds.append((rows, tuple(int(v) for v in a)))
    return G.astype(int), rounds
def mi(v): return str(v).replace("-", "−")
def design(lang):
    t = tr(lang); G, rounds = trace(); o = ""
    assert [r[1] for r in rounds[0][0]] == [0, 1, 2, 1] and rounds[2][1] == (1, 1, 1, 1) and sum(r[3] for rr, _ in rounds for r in rr) == 4
    # --- left top: the plane
    o += card(0, 8, 350, 226, "blue") + head(0, 8, 350, "blue", t("XOR 型：沒有直線能分開", "XOR type: no line separates them"), 34, 17)
    cx, cy, u = 175, 118, 46
    s = f'<line x1="{cx-1.45*u:.0f}" y1="{cy}" x2="{cx+1.45*u:.0f}" y2="{cy}" stroke="#6b645a" stroke-width="2"/><line x1="{cx}" y1="{cy+1.2*u:.0f}" x2="{cx}" y2="{cy-1.2*u:.0f}" stroke="#6b645a" stroke-width="2"/>'
    for sg in (1, -1): s += f'<line x1="{cx-sg*1.2*u:.0f}" y1="{cy-1.2*u:.0f}" x2="{cx+sg*1.2*u:.0f}" y2="{cy+1.2*u:.0f}" stroke="#e0c85f" stroke-width="2.5" stroke-dasharray="8 6"/>'
    for (px, py) in ((1, 0), (-1, 0)): s += f'<circle cx="{cx+px*u}" cy="{cy-py*u}" r="15" fill="#47688f"/>'
    for (px, py) in ((0, 1), (0, -1)): s += f'<rect x="{cx+px*u-13}" y="{cy-py*u-13}" width="26" height="26" rx="4" fill="#94503a"/>'
    o += svg(0, 0, W, H, s)
    for n_, (px, py) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1)), 1):
        o += txt(cx + px * u - 14, cy - py * u - 12, 28, str(n_), 17, "#ffffff", True, "center", 24)
    o += txt(6, 182, 338, t("藍圓 +1，橘方 −1（數字是點號）<br>虛線分開左右（g > 0）與上下（g < 0）", "blue +1, orange −1; numbers = point ids<br>dashes: left/right g > 0, up/down g < 0"), 16, MUT, False, "center", 48)
    # --- left bottom: kernel matrix
    o += card(0, 244, 350, 164, "orange") + head(0, 244, 350, "orange", t("核矩陣 G（K = (xᵀz+1)²）", "Kernel matrix G (K = (xᵀz+1)²)"), 34, 17)
    for i in range(4):
        o += txt(14, 284 + i * 30, 26, str(i + 1), 16, MUT, True, "center", 24)
        for j in range(4):
            v = G[i, j]; role = "orange" if v == 4 else ("grey" if v == 0 else "yellow")
            o += chip(46 + j * 74, 283 + i * 30, 66, role, str(v), False, 26, 17)
    # --- right: round-by-round
    o += card(370, 8, 790, 316, "green") + head(370, 8, 790, "green", t("逐輪追蹤（紅：分錯，α 的該點加 1；綠：分對）", "Round by round (red: a mistake adds 1 to that point's α; green: correct)"), 34, 17)
    for r, (rows, al) in enumerate(rounds):
        x = 382 + r * 260
        o += chip(x, 52, 240, "green", t(f"第 {r+1} 輪", f"Round {r+1}"), True, 30, 17)
        for i, (pt, g, yg, bad) in enumerate(rows):
            y0 = 90 + i * 44; role = "red" if bad else "green"
            o += card(x, y0, 240, 38, role, .2 if bad else .1)
            o += txt(x + 8, y0 + 7, 176, (f"點{pt}　g={mi(g)}，y·g={mi(yg)}" if lang == "zh" else f"pt {pt}  g={mi(g)}, y·g={mi(yg)}"), 16, TXT, False, "left", 24)
            o += chip(x + 192, y0 + 5, 42, "red" if bad else "green", t("錯", "bad") if bad else t("對", "ok"), True, 28, 16)
        o += txt(x, 270 + 4, 240, "α = (" + ", ".join(str(a_) for a_ in al) + ")" + (t("　停", "  stop") if r == 2 else ""), 19, TXT, True, "center", 30)
    # --- right bottom: one check
    o += card(370, 332, 790, 76, "yellow", .12)
    o += txt(384, 338, 560, t("驗算：第 1 輪點 3，α=(1,0,0,0)", "Check: round 1, point 3, α=(1,0,0,0)"), 16, MUT, True, "left", 24)
    o += formula(380, 364, 560, r"g(x_3)=\alpha_1y_1K(x_1,x_3)+\sum_j\alpha_jy_j=1\cdot1\cdot1+1=2", 19, 40)
    o += formula(975, 364, 170, r"y_3g=-2\leq 0", 19, 40) + txt(975, 336, 170, t("分錯", "mistake"), 16, RD, True, "center", 24)
    # --- bottom takeaway
    o += card(0, 418, 1160, 50, "yellow", .14)
    o += txt(18, 430, 420, t("分錯 4 次，所以 α = (1, 1, 1, 1)；最後", "4 mistakes, so α = (1, 1, 1, 1); finally"), 18, TXT, True, "left", 26)
    o += formula(430 if lang == "zh" else 520, 424, 300, r"g(x)=2\,(x_1^2-x_2^2)", 22, 38)
    o += txt(740 if lang == "zh" else 810, 430, 410 if lang == "zh" else 340, t("(2, 0) 得 +8，(0, 2) 得 −8", "(2, 0) gives +8, (0, 2) gives −8"), 18, TXT, True, "left", 26)
    return o
ALT_ZH = "手算追蹤：四個點 x1=(1,0)、x2=(−1,0) 為 +1，x3=(0,1)、x4=(0,−1) 為 −1，核 K=(xᵀz+1)²，XOR 型資料沒有直線能分開。核矩陣 G：第一列 4、0、1、1；第二列 0、4、1、1；第三列 1、1、4、0；第四列 1、1、0、4。第 1 輪：點 1 g=0 錯、點 2 g=1 對、點 3 g=2 錯、點 4 g=1 錯，α=(1,0,1,1)；第 2 輪：點 1 g=1 對、點 2 g=−3 錯、點 3、4 對，α=(1,1,1,1)；第 3 輪 g 為 2、2、−2、−2 全對，停。驗算：第 1 輪點 3 時 g=2、y·g=−2，分錯。分錯共 4 次，α=(1,1,1,1)，最後 g(x)=2(x1²−x2²)：(2,0) 得 +8，(0,2) 得 −8。"
ALT_EN = "Hand trace: the four points x1=(1,0), x2=(−1,0) are +1 and x3=(0,1), x4=(0,−1) are −1 with kernel K=(xᵀz+1)²; this XOR-type data cannot be separated by a line. The kernel matrix G has rows 4,0,1,1; 0,4,1,1; 1,1,4,0; 1,1,0,4. Round 1: point 1 g=0 wrong, point 2 g=1 right, point 3 g=2 wrong, point 4 g=1 wrong, α=(1,0,1,1); round 2: point 1 g=1 right, point 2 g=−3 wrong, points 3 and 4 right, α=(1,1,1,1); round 3 g=2,2,−2,−2 all right, stop. Check: round 1 point 3 gives g=2 and y·g=−2, a mistake. Four mistakes in total, α=(1,1,1,1), and the final g(x)=2(x1²−x2²): (2,0) gives +8, (0,2) gives −8."
def native_fig(jsondir):
    return native("kp20_trace", "lec05-kp20-trace", W, H, jsondir, ALT_EN if os.environ.get("KP_LANG") == "en" else ALT_ZH)
if __name__ == "__main__":
    build("kp20_trace", design, W, H, sys.argv[1] if len(sys.argv) > 1 else ".")
