"""KP01: why deep learning uses first-order optimisation — Explainer sheet (mechanism pane + evidence pane).

Every number is computed here. Toy problem f(x,y) = 1/2 (x^2 + 25 y^2), theta0 = (10, 2), eta = 0.06 (< 2/25 = 0.08, the stability limit).
Run:  py assets/figures/src/idl-kp1-first-order.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import MONO, T, box, footer, header, line, page, r1, text, write_outputs  # noqa: E402

SLUG = "idl-kp1-first-order"
LAM = (1.0, 25.0)
ETA = 0.06
TH0 = (10.0, 2.0)
K = 30


def f(x, y):
    return 0.5 * (LAM[0] * x * x + LAM[1] * y * y)


# ---------------------------------------------------------------- computations
path = [TH0]
for _ in range(K):
    x, y = path[-1]
    path.append((x - ETA * LAM[0] * x, y - ETA * LAM[1] * y))
fvals = [f(*p) for p in path]
newton = (TH0[0] - TH0[0], TH0[1] - TH0[1])          # H^-1 g = theta for a quadratic with minimum at 0
assert abs(f(*newton)) < 1e-12
kappa = LAM[1] / LAM[0]
eta_max = 2 / LAM[1]


def human_bytes(b):
    for unit, s in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= s:
            return f"{b / s:.3g} {unit}"
    return f"{b:.0f} B"


def human_time(sec):
    if sec < 1e-3:
        return f"{sec * 1e6:.1f} µs"
    if sec < 60:
        return f"{sec:.2g} s"
    if sec < 86400:
        return f"{sec / 3600:.3g} 小時 h"
    return f"{sec / 86400:.3g} 天 d"


FLOPS = 10e12
rows = []
for d in (10**2, 10**4, 10**6):
    rows.append(dict(d=d, grad=human_bytes(4 * d), hess=human_bytes(4 * d * d), t=human_time(d**3 / FLOPS)))
print(f"kappa={kappa}  eta_max={eta_max}  f0={fvals[0]:.1f}  f10={fvals[10]:.2f}  f30={fvals[30]:.3f}  newton f1=0")
for r in rows:
    print(r)

# ---------------------------------------------------------------- left: mechanism pane
L = []
L.append(text(40, 160, "機制 Mechanism", 16, 600))
L.append(box(40, 176, 440, 58, T["paper"], T["ink"], 1))
L.append(text(56, 200, "問題 Problem", 16, 600))
L.append(text(56, 222, "最小化 L(θ)，θ 有 d 個參數", 12, 400, T["muted"]))
L.append(text(464, 200, "min L(θ)", 14, 500, T["muted"], "end", MONO))
for x0, focal, ttl, formula, l1, l2 in (
    (40, True, "一階 First-order", "θ ← θ − η g", "只算梯度 g", "記憶體 O(d)"),
    (270, False, "二階 Second-order", "θ ← θ − H⁻¹ g", "還要 Hessian H", "記憶體 O(d²)"),
):
    L.append(line(x0 + 105, 234, x0 + 105, 268, T["accent"] if focal else T["muted"], 1.4, marker="arrow-accent" if focal else "arrow"))
    L.append(box(x0, 274, 210, 150, T["accent_tint"] if focal else T["paper"], T["accent"] if focal else T["ink"], 1.4 if focal else 1))
    L.append(text(x0 + 14, 302, ttl, 16, 600))
    L.append(text(x0 + 14, 332, formula, 14, 500, T["ink"], family=MONO))
    L.append(text(x0 + 14, 366, l1, 14, 400))
    L.append(text(x0 + 14, 390, l2, 14, 400))
    L.append(text(x0 + 14, 412, "First-order" if focal else "Needs the Hessian", 12, 400, T["muted"]))
    L.append(line(x0 + 105, 424, x0 + 105, 462, T["accent"] if focal else T["muted"], 1.4, marker="arrow-accent" if focal else "arrow"))
L.append(box(40, 468, 440, 96, T["accent_tint"], T["accent"], 1.4))
L.append(text(56, 498, "深度學習用一階", 16, 600))
L.append(text(56, 520, "d 常達百萬以上，O(d²) 的記憶體放不下。", 14, 400))
L.append(text(56, 546, "At d ≥ 10⁶ the O(d²) Hessian does not fit in memory.", 12, 400, T["muted"]))
L.append(text(40, 592, "代價：每步較「慢」，要靠步長 η 與調整技巧補。", 14, 400))
L.append(text(40, 612, "Price: more, slower steps — η and tricks make up for it.", 12, 400, T["muted"]))

# ---------------------------------------------------------------- right: evidence pane
R = []
R.append(line(494, 150, 494, 640, T["rule"], 1))
# panel 1: contour plot
PX0, PY0, PW, PH = 520, 190, 340, 170
xr, yr = (-1.0, 11.0), (-2.6, 2.6)


def sx(x):
    return PX0 + (x - xr[0]) / (xr[1] - xr[0]) * PW


def sy(y):
    return PY0 + (yr[1] - y) / (yr[1] - yr[0]) * PH


R.append(text(520, 160, "證據 1　同一個問題走兩種路", 16, 600))
R.append(text(520, 178, "Evidence 1: two ways down the same bowl", 12, 400, T["muted"]))
R.append(f'<clipPath id="plotclip"><rect x="{PX0}" y="{PY0}" width="{PW}" height="{PH}"/></clipPath>')
R.append(f'<rect x="{PX0}" y="{PY0}" width="{PW}" height="{PH}" fill="{T["paper2"]}" stroke="{T["rule_solid"]}" stroke-width="1"/>')
levels = (100, 60, 30, 12, 4)
for c in levels:
    a = math.sqrt(2 * c / LAM[0])
    b = math.sqrt(2 * c / LAM[1])
    R.append(f'<ellipse cx="{r1(sx(0))}" cy="{r1(sy(0))}" rx="{r1(a / (xr[1] - xr[0]) * PW)}" ry="{r1(b / (yr[1] - yr[0]) * PH)}" '
             f'fill="none" stroke="{T["muted"]}" stroke-opacity="0.45" stroke-width="1" clip-path="url(#plotclip)"/>')
pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in path[:13])
R.append(f'<polyline points="{pts}" fill="none" stroke="{T["accent"]}" stroke-width="1.6"/>')
for x, y in path[:13]:
    R.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{T["accent"]}"/>')
R.append(line(sx(TH0[0]), sy(TH0[1]), sx(0) + 5, sy(0) - 1.5, T["link"], 1.6, "6,4", "arrow-link"))
R.append(f'<circle cx="{sx(TH0[0]):.1f}" cy="{sy(TH0[1]):.1f}" r="4" fill="{T["ink"]}"/>')
R.append(text(sx(TH0[0]) - 8, sy(TH0[1]) - 8, "θ₀ = (10, 2)", 12, 500, T["ink"], "end", MONO))
R.append(text(sx(0) + 8, sy(0) + 20, "最小點 minimum", 12, 400, T["muted"]))
R.append(line(PX0 + 8, PY0 + PH + 18, PX0 + 36, PY0 + PH + 18, T["accent"], 1.6))
R.append(text(PX0 + 44, PY0 + PH + 22, f"一階 GD，η = {ETA}，之字形、慢", 12, 400))
R.append(line(PX0 + 8, PY0 + PH + 38, PX0 + 36, PY0 + PH + 38, T["link"], 1.6, "6,4"))
R.append(text(PX0 + 44, PY0 + PH + 42, "二階 Newton，1 步到底（二次函數）", 12, 400))

# panel 2: cost strip
CX0 = 890
R.append(text(CX0, 160, "證據 2　二階一步的帳單", 16, 600))
R.append(text(CX0, 178, "Evidence 2: the bill for one second-order step", 12, 400, T["muted"]))
cols = [CX0, CX0 + 70, CX0 + 160, CX0 + 262]
hdr = ["參數 d", "一階 g", "二階 H", "H⁻¹ 時間 d³"]
for cx, h in zip(cols, hdr):
    R.append(text(cx, 206, h, 12, 600, T["muted"]))
R.append(line(CX0, 214, 1240, 214, T["rule_solid"], 1))
for i, r in enumerate(rows):
    y = 240 + i * 38
    d_txt = f"10{'²⁴⁶'[i]}"
    last = i == len(rows) - 1
    R.append(text(cols[0], y, d_txt, 14, 500, T["ink"], family=MONO))
    R.append(text(cols[1], y, r["grad"], 14, 400, T["ink"], family=MONO))
    R.append(text(cols[2], y, r["hess"], 14, 600 if last else 400, T["accent"] if last else T["ink"], family=MONO))
    R.append(text(cols[3], y, r["t"], 14, 600 if last else 400, T["accent"] if last else T["ink"], family=MONO))
R.append(text(CX0, 362, "float32；時間以 10 TFLOPS 估算 d³ 次運算", 12, 400, T["muted"]))
R.append(text(CX0, 380, "float32 values; time assumes d³ ops at 10 TFLOPS", 12, 400, T["soft"]))
R.append(text(CX0, 404, "d = 10⁶ 時 H 要 4 TB，反矩陣約 1 天以上", 14, 500))

# panel 3: convergence on a log scale
SX0, SY0, SW_, SH_ = 520, 468, 520, 120
lo, hi = -0.2, 2.1


def gx(k):
    return SX0 + k / K * SW_


def gy(v):
    return SY0 + (hi - math.log10(v)) / (hi - lo) * SH_


R.append(text(520, 446, "證據 3　一階的代價：損失下降得慢（對數刻度）", 16, 600))
R.append(text(520, 463, "Evidence 3: the first-order price — slow decrease (log scale)", 12, 400, T["muted"]))
R.append(f'<rect x="{SX0}" y="{SY0}" width="{SW_}" height="{SH_}" fill="{T["paper2"]}" stroke="{T["rule_solid"]}" stroke-width="1"/>')
for v in (1, 10, 100):
    R.append(line(SX0, gy(v), SX0 + SW_, gy(v), T["rule"], 1))
    R.append(text(SX0 - 6, gy(v) + 4, str(v), 12, 400, T["muted"], "end", MONO))
R.append(f'<polyline points="{" ".join(f"{gx(k):.1f},{gy(v):.1f}" for k, v in enumerate(fvals))}" fill="none" stroke="{T["accent"]}" stroke-width="1.8"/>')
for k in (0, 10, 30):
    R.append(f'<circle cx="{gx(k):.1f}" cy="{gy(fvals[k]):.1f}" r="3.2" fill="{T["accent"]}"/>')
R.append(text(gx(10) + 6, gy(fvals[10]) - 8, f"{fvals[10]:.1f}", 12, 500, T["accent"], family=MONO))
R.append(text(gx(30) - 6, gy(fvals[30]) - 8, f"{fvals[30]:.1f}", 12, 500, T["accent"], "end", MONO))
R.append(f'<circle cx="{gx(1):.1f}" cy="{SY0 + SH_ - 4}" r="3.6" fill="{T["link"]}"/>')
R.append(text(gx(1) + 10, SY0 + SH_ - 12, "Newton：第 1 步 f = 0", 12, 500, T["link"]))
for k in (0, 10, 20, 30):
    R.append(text(gx(k), SY0 + SH_ + 16, str(k), 12, 400, T["muted"], "middle", MONO))
R.append(text(SX0 + SW_ + 40, SY0 + SH_ + 16, "步 step", 12, 400, T["muted"]))
R.append(text(1060, 490, f"條件數 κ = {LAM[1]:.0f}", 14, 600, T["ink"]))
R.append(text(1060, 512, f"η 必須 < 2/25 = {eta_max:.2f}", 14, 400))
R.append(text(1060, 534, f"否則 y 方向發散", 14, 400))
R.append(text(1060, 560, "Stability needs η < 2/λmax;", 12, 400, T["muted"]))
R.append(text(1060, 577, "the flat x-direction then crawls.", 12, 400, T["muted"]))

body = "".join([
    header("為什麼深度學習用一階最佳化", "Why deep learning optimises with first-order methods", "κ = 25 · η = 0.06"),
    "".join(L), "".join(R),
    footer("橘＝一階（梯度下降）；藍虛線＝二階（Newton）；等高線＝損失相同的點。",
           "Orange: first-order (gradient descent). Blue dashed: second-order (Newton). Contours: equal loss.",
           "玩具二次函數；真實網路非凸，Newton 不保證一步到底。",
           "A toy quadratic. Real networks are non-convex, so Newton is not one-step there."),
])
html_text = page(SLUG, "為什麼用一階最佳化 Why first-order",
                 "Explainer sheet: on the quadratic f = (x^2 + 25 y^2)/2, gradient descent zigzags and slows while Newton lands in one step, but the Hessian costs O(d^2) memory and O(d^3) time.",
                 body)

if __name__ == "__main__":
    write_outputs(SLUG, html_text)
