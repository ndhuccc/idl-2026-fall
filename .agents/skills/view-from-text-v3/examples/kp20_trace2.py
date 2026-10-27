"""KP20 page 5 redone as two pages (view-from-text-v3, JianBauDau fill effects + polygons).
 A: data, kernel matrix, the rule, one worked cell.   B: the 12-step trace with the alpha vector, the final discriminant and its regions."""
import os, sys, math
import numpy as np
SK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, SK + "/view-from-text-v3/tools"); sys.path.insert(0, SK + "/view-from-text-v3/examples")
from kp17_pages import txt, named, card, head, chip, formula, svg, tr, hexc, TXT, MUT, INK, build, native
BL, OR, RD, GR, YE = (hexc(c) for c in ("blue", "red", "orange", "green", "yellow")); BL, RD, OR, GR, YE = hexc("blue"), hexc("red"), hexc("orange"), hexc("green"), hexc("yellow")
W, H = 1160, 480

def trace():
    P = np.array([[1, 0], [-1, 0], [0, 1], [0, -1.]]); y = np.array([1, 1, -1, -1.]); G = (P @ P.T + 1) ** 2; a = np.zeros(4); rounds = []
    for ep in range(3):
        rows = []
        for i in range(4):
            g = (a * y) @ G[:, i] + (a * y).sum(); bad = y[i] * g <= 0
            if bad: a[i] += 1
            rows.append((i + 1, int(round(g)), int(round(y[i] * g)), bool(bad), tuple(int(v) for v in a)))
        rounds.append(rows)
    return G.astype(int), rounds
def mi(v): return str(v).replace("-", "−")

def design_a(lang):
    t = tr(lang); G, rounds = trace(); o = ""
    # ① data
    o += card(0, 0, 376, 304, "blue") + head(0, 0, 376, "blue", t("① 資料：XOR 型，沒有直線分得開", "① Data: XOR type, no line separates them"), 36, 17)
    cx, cy, u = 188, 160, 74
    s = f'<line x1="{cx-1.5*u:.0f}" y1="{cy}" x2="{cx+1.5*u:.0f}" y2="{cy}" stroke="#6b645a" stroke-width="2"/><line x1="{cx}" y1="{cy+1.4*u:.0f}" x2="{cx}" y2="{cy-1.4*u:.0f}" stroke="#6b645a" stroke-width="2"/>'
    for (px, py) in ((1, 0), (-1, 0)): s += f'<circle cx="{cx+px*u}" cy="{cy-py*u}" r="19" fill="#47688f"/>'
    for (px, py) in ((0, 1), (0, -1)): s += f'<rect x="{cx+px*u-17}" y="{cy-py*u-17}" width="34" height="34" rx="5" fill="#94503a"/>'
    o += svg(0, 0, W, H, s)
    for n_, (px, py) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1)), 1):
        o += txt(cx + px * u - 14, cy - py * u - 12, 28, str(n_), 18, "#ffffff", True, "center", 24)
    for lab, (px, py), (dx, dy) in (("(1, 0)", (1, 0), (-24, 26)), ("(−1, 0)", (-1, 0), (-34, 26)), ("(0, 1)", (0, 1), (26, -12)), ("(0, −1)", (0, -1), (26, -12))):
        o += txt(cx + px * u + dx, cy - py * u + dy, 70, lab, 16, MUT, False, "left", 24)
    o += txt(8, 262, 360, t("點 1、2 屬類別 +1（藍圓）<br>點 3、4 屬類別 −1（橘方）", "points 1, 2: class +1 (blue circles)<br>points 3, 4: class −1 (orange squares)"), 16, TXT, False, "center", 48)
    # ② kernel matrix
    o += card(392, 0, 376, 304, "orange") + head(392, 0, 376, "orange", t("② 核矩陣 G：Gij = K(xi, xj)", "② Kernel matrix G: Gij = K(xi, xj)"), 36, 17)
    o += formula(396, 38, 368, r"K(x,z)=(x^{T}z+1)^2", 19, 36)
    for j in range(4): o += txt(470 + j * 72, 78, 60, f"x{j+1}", 16, MUT, True, "center", 22)
    for i in range(4):
        o += txt(410, 104 + i * 40, 56, f"x{i+1}", 16, MUT, True, "center", 22)
        for j in range(4):
            v = G[i, j]; role = "orange" if v == 4 else ("grey" if v == 0 else "yellow")
            o += chip(470 + j * 72, 102 + i * 40, 60, role, str(v), False, 34, 18)
    o += f'<div style="position:absolute;left:464px;top:98px;width:296px;height:42px;border:3px dashed {GR};border-radius:10px"></div>'
    o += txt(404, 270, 352, t("綠框：第 1 列；灰 0、黃 1、橘 4", "green box: row 1; grey 0, yellow 1, orange 4"), 16, MUT, False, "left", 24)
    # ③ the rule
    o += card(784, 0, 376, 304, "green") + head(784, 0, 376, "green", t("③ 規則：看一點、算 g、錯了加 1", "③ The rule: check, compute g, add 1"), 36, 17)
    o += txt(796, 46, 352, t("αi：第 i 點被分錯的次數，從 0 開始", "αi: how many times point i was misclassified, from 0"), 16, TXT, False, "left", 48)
    o += formula(792, 100, 360, r"g(x_i)=\sum_j\alpha_jy_jG_{ji}+\sum_j\alpha_jy_j", 20, 48)
    o += chip(796, 158, 352, "red", t("yi·g(xi) ≤ 0：分錯 → αi ← αi + 1", "yi·g(xi) ≤ 0: mistake → αi ← αi + 1"), False, 36, 17)
    o += chip(796, 202, 352, "green", t("yi·g(xi) > 0：分對 → α 不變", "yi·g(xi) > 0: correct → α unchanged"), False, 36, 17)
    o += txt(796, 252, 352, t("一輪一輪掃過四個點，直到整輪都沒分錯為止", "sweep the four points round after round until a whole round has no mistake"), 16, MUT, False, "left", 48)
    # ④ worked cell
    o += card(0, 316, 1160, 164, "yellow", .12) + head(0, 316, 1160, "yellow", t("④ 驗算一格：第 1 輪、點 3（此時 α = (1, 0, 0, 0)）", "④ Check one cell: round 1, point 3 (now α = (1, 0, 0, 0))"), 36, 17)
    o += formula(14, 362, 1130, r"g(x_3)=\alpha_1y_1K(x_1,x_3)+\sum_j\alpha_jy_j=1\cdot1\cdot K(x_1,x_3)+1=1\cdot1\cdot1+1=2", 22, 44)
    o += formula(14, 410, 560, r"y_3\,g(x_3)=(-1)(2)=-2\leq 0", 22, 40)
    o += chip(600, 414, 260, "red", t("分錯 → α₃ ← 0 + 1 = 1", "mistake → α3 ← 0 + 1 = 1"), True, 38, 18)
    o += txt(876, 418, 280, t("K(x₁,x₃) = (0+1)² = 1", "K(x1,x3) = (0+1)² = 1"), 17, TXT, True, "left", 30)
    o += txt(14, 450, 1130, t("偏置項 Σαⱼyⱼ 就是 w₀，同樣由計數器得出：所以權重向量從頭到尾都不必算。", "The bias term Σαjyj is w0 and also comes from the counters: the weight vector is never computed."), 16, MUT, False, "left", 24)
    return o

def design_b(lang):
    t = tr(lang); G, rounds = trace(); o = ""
    o += card(0, 0, 764, 480, "green") + head(0, 0, 764, "green", t("逐步追蹤：看一點、算 g，分錯就 αi 加 1", "Step by step: look at a point, compute g, a mistake adds 1 to αi"), 36, 17)
    cols = {"r": (12, 40), "pt": (54, 30), "g": (96, 70), "yg": (176, 70), "res": (256, 84)}; AX = 392
    for k_, lab in (("r", t("輪", "rd")), ("pt", t("點", "pt")), ("g", "g(xi)"), ("yg", "yi·g"), ("res", t("結果", "result"))): o += txt(cols[k_][0], 42, cols[k_][1], lab, 16, MUT, True, "center", 22)
    for j in range(4): o += txt(AX + j * 82, 42, 70, f"α{j+1}", 16, MUT, True, "center", 22)
    y0 = 66; prev = (0, 0, 0, 0)
    for r, rows in enumerate(rounds):
        for i, (pt, g, yg, bad, al) in enumerate(rows):
            y = y0 + (r * 4 + i) * 29 + r * 8
            o += card(10, y, 744, 27, "red" if bad else "green", .2 if bad else .08)
            if i == 0: o += txt(cols["r"][0], y + 1, cols["r"][1], str(r + 1), 17, TXT, True, "center", 25)
            o += chip(cols["pt"][0], y + 2, 30, "grey", str(pt), False, 23, 16)
            o += txt(cols["g"][0], y + 1, cols["g"][1], mi(g), 17, TXT, False, "center", 25) + txt(cols["yg"][0], y + 1, cols["yg"][1], mi(yg), 17, RD if bad else GR, True, "center", 25)
            o += txt(cols["res"][0], y + 1, cols["res"][1], t("✗ 錯", "✗ bad") if bad else t("✓ 對", "✓ ok"), 16, RD if bad else GR, True, "center", 25)
            for j in range(4):
                changed = al[j] != prev[j]
                o += chip(AX + j * 82, y + 2, 70, "orange" if changed else "grey", str(al[j]), changed, 23, 17)
            prev = al
    yb = y0 + 12 * 29 + 2 * 8 + 6
    o += card(10, yb, 744, 36, "green", .22) + txt(18, yb + 5, 728, t("第 3 輪整輪全對（yi·g 全為 2）→ 停，α = (1, 1, 1, 1)", "Round 3 is all correct (every yi·g is 2) → stop, α = (1, 1, 1, 1)"), 18, TXT, True, "center", 26)
    # right: final discriminant + regions
    o += card(780, 0, 380, 150, "violet") + head(780, 0, 380, "violet", t("最後的判別式", "The final discriminant"), 36, 17)
    o += formula(786, 42, 368, r"g(x)=2\,(x_1^2-x_2^2)", 26, 48)
    o += txt(792, 98, 356, t("g(2, 0) = +8（類別 +1）<br>g(0, 2) = −8（類別 −1）", "g(2, 0) = +8 (class +1)<br>g(0, 2) = −8 (class −1)"), 17, TXT, True, "center", 48)
    o += card(780, 162, 380, 318, "blue") + head(780, 162, 380, "blue", t("它在平面上的分區", "Its regions in the plane"), 36, 17)
    cx, cy, h_ = 970, 308, 100; xl, xr, yt, yb2 = cx - h_, cx + h_, cy - h_, cy + h_
    s = f'<defs><linearGradient id="gA" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#6a9fd6" stop-opacity="0.75"/><stop offset="1" stop-color="#6a9fd6" stop-opacity="0.35"/></linearGradient><linearGradient id="gB" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#6a9fd6" stop-opacity="0.35"/><stop offset="1" stop-color="#6a9fd6" stop-opacity="0.75"/></linearGradient><linearGradient id="gC" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d97757" stop-opacity="0.75"/><stop offset="1" stop-color="#d97757" stop-opacity="0.35"/></linearGradient><linearGradient id="gD" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d97757" stop-opacity="0.35"/><stop offset="1" stop-color="#d97757" stop-opacity="0.75"/></linearGradient></defs>'
    s += f'<polygon points="{cx},{cy} {xl},{yt} {xl},{yb2}" fill="url(#gB)" stroke="#6a9fd6" stroke-width="1.5"/><polygon points="{cx},{cy} {xr},{yt} {xr},{yb2}" fill="url(#gA)" stroke="#6a9fd6" stroke-width="1.5"/>'
    s += f'<polygon points="{cx},{cy} {xl},{yt} {xr},{yt}" fill="url(#gC)" stroke="#d97757" stroke-width="1.5"/><polygon points="{cx},{cy} {xl},{yb2} {xr},{yb2}" fill="url(#gD)" stroke="#d97757" stroke-width="1.5"/>'
    s += f'<line x1="{xl-6}" y1="{cy}" x2="{xr+6}" y2="{cy}" stroke="#b7b0a5" stroke-width="1.5"/><line x1="{cx}" y1="{yb2+6}" x2="{cx}" y2="{yt-6}" stroke="#b7b0a5" stroke-width="1.5"/>'
    for (px, py) in ((1, 0), (-1, 0)): s += f'<circle cx="{cx+px*68}" cy="{cy-py*68}" r="17" fill="#47688f" stroke="#ffffff" stroke-width="2"/>'
    for (px, py) in ((0, 1), (0, -1)): s += f'<rect x="{cx+px*68-15}" y="{cy-py*68-15}" width="30" height="30" rx="4" fill="#94503a" stroke="#ffffff" stroke-width="2"/>'
    o += svg(0, 0, W, H, s)
    for n_, (px, py) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1)), 1): o += txt(cx + px * 68 - 14, cy - py * 68 - 12, 28, str(n_), 17, "#ffffff", True, "center", 24)
    o += txt(xr - 62, cy + 34, 58, "g > 0", 17, TXT, True, "right", 24) + txt(xl + 4, cy + 34, 58, "g > 0", 17, TXT, True, "left", 24)
    o += txt(cx + 14, yt + 6, 58, "g < 0", 17, TXT, True, "left", 24) + txt(cx + 14, yb2 - 30, 58, "g < 0", 17, TXT, True, "left", 24)
    o += txt(790, yb2 + 12, 360, t("邊界是兩條斜線 |x₁| = |x₂|，直線做不到", "Boundary: two slanted lines<br>|x1| = |x2|; no straight line can"), 16, TXT, True, "center", 48)
    return o

ALT_A_ZH = "手算追蹤（一）：四個點 x1=(1,0)、x2=(−1,0) 為 +1，x3=(0,1)、x4=(0,−1) 為 −1，XOR 型資料沒有直線能分開。核 K=(xᵀz+1)² 的核矩陣 G：第一列 4、0、1、1；第二列 0、4、1、1；第三列 1、1、4、0；第四列 1、1、0、4。規則：αi 是第 i 點被分錯的次數；g(xi)=Σαjyj Gji+Σαjyj；yi·g≤0 為分錯，αi 加 1。驗算：第 1 輪點 3、α=(1,0,0,0)，g=1·1·1+1=2，y3·g=−2≤0，分錯，α3 變 1。"
ALT_A_EN = "Hand trace (1): points x1=(1,0), x2=(−1,0) are +1 and x3=(0,1), x4=(0,−1) are −1; XOR-type data no line can separate. Kernel K=(xᵀz+1)² gives the kernel matrix G with rows 4,0,1,1; 0,4,1,1; 1,1,4,0; 1,1,0,4. The rule: αi counts the mistakes on point i; g(xi)=Σαjyj Gji+Σαjyj; yi·g≤0 is a mistake and adds 1 to αi. Check: round 1, point 3, α=(1,0,0,0): g=1·1·1+1=2, y3·g=−2≤0, a mistake, α3 becomes 1."
ALT_B_ZH = "手算追蹤（二）：第 1 輪：點 1 g=0 錯 α=(1,0,0,0)、點 2 g=1 對、點 3 g=2 錯 α=(1,0,1,0)、點 4 g=1 錯 α=(1,0,1,1)；第 2 輪：點 1 g=1 對、點 2 g=−3 錯 α=(1,1,1,1)、點 3、4 g=−2 對；第 3 輪：g=2、2、−2、−2 全對，停，α=(1,1,1,1)。最後 g(x)=2(x1²−x2²)：g(2,0)=+8（+1 類）、g(0,2)=−8（−1 類）；分區邊界是兩條斜線 |x1|=|x2|，左右 g>0、上下 g<0，直線做不到。"
ALT_B_EN = "Hand trace (2): round 1: point 1 g=0 bad α=(1,0,0,0), point 2 g=1 ok, point 3 g=2 bad α=(1,0,1,0), point 4 g=1 bad α=(1,0,1,1); round 2: point 1 g=1 ok, point 2 g=−3 bad α=(1,1,1,1), points 3 and 4 g=−2 ok; round 3: g=2,2,−2,−2 all ok, stop, α=(1,1,1,1). Final g(x)=2(x1²−x2²): g(2,0)=+8 (class +1), g(0,2)=−8 (class −1); the boundary is two slanted lines |x1|=|x2| with g>0 left/right and g<0 up/down, which one straight line cannot do."
def native_a(jsondir): return native("kp20_trace_a", "lec05-kp20-trace", W, H, jsondir, ALT_A_EN if os.environ.get("KP_LANG") == "en" else ALT_A_ZH)
def native_b(jsondir): return native("kp20_trace_b", "lec05-kp20-trace2", W, H, jsondir, ALT_B_EN if os.environ.get("KP_LANG") == "en" else ALT_B_ZH)
if __name__ == "__main__":
    out = sys.argv[1]; build("kp20_trace_a", design_a, W, H, out); build("kp20_trace_b", design_b, W, H, out)
