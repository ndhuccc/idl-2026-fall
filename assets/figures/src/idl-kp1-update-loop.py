"""KP01 (Ch03 backpropagation & optimization): one update step as a Loop, built with the diagram-design skill.

Type: Loop (references/type-loop.md) on the dark project skin, slide-16x9 (viewBox 1280x720), bilingual (Chinese leads).
Geometry is parametric (type-loop.md §2): station k sits on a circle of radius R at -90deg + k*72deg, ring arrows are
circular arcs between adjacent boxes, dashed spokes run radially from each station to the shared hub.
Run:  py assets/figures/src/idl-kp1-update-loop.py        (needs: playwright for the PNG)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_common import MONO, SANS, T, box, esc, footer, fx, header, line, page, r1, text, write_outputs  # noqa: E402

SLUG = "idl-kp1-update-loop"
N, R = 5, 204
CX, CY = 640, 396
SW, SH = 180, 80           # station box
HW, HH = 120, 72           # hub box
STROKE_RING = 1.4
GAP = 6                    # marker gap before the hub


def inside(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


stations = [
    dict(zh="預測", en="Predict", sub="ŷ = f(x; θ)", tag="假設類別"),
    dict(zh="損失", en="Loss", sub="L = loss(ŷ, y)", tag="損失", focal=True),
    dict(zh="梯度", en="Gradient", sub="g = ∇L(θ)", tag="優化器"),
    dict(zh="步長", en="Step", sub="d = −η · g", tag="優化器"),
    dict(zh="更新", en="Update", sub="θ ← θ + d", tag="優化器"),
]
for k, s in enumerate(stations):
    th = math.radians(-90 + k * 360 / N)
    s["theta"] = th
    s["u"] = (math.cos(th), math.sin(th))
    s["c"] = (CX + R * math.cos(th), CY + R * math.sin(th))
    s["rect"] = (round(s["c"][0] - SW / 2, 3), round(s["c"][1] - SH / 2, 3), SW, SH)


def arc_interval(rect, theta):
    """Angles (rad) at which the ring circle enters / leaves the box, searched around the station angle."""
    step = math.radians(0.02)
    lo = hi = None
    a = theta - math.radians(40)
    while a <= theta + math.radians(40):
        if inside(CX + R * math.cos(a), CY + R * math.sin(a), rect):
            lo = a if lo is None else lo
            hi = a
        a += step
    assert lo is not None and hi is not None
    return lo, hi


for s in stations:
    s["entry"], s["exit"] = arc_interval(s["rect"], s["theta"])

# ------------------------------------------------------------------ connectors (drawn before the boxes)
ring, spokes = [], []
overhang = STROKE_RING * 1.0
for k, s in enumerate(stations):
    j = (k + 1) % N
    d = stations[j]
    x1, y1 = CX + R * math.cos(s["exit"]), CY + R * math.sin(s["exit"])
    phi_end = d["entry"] - overhang / R
    if phi_end < s["exit"]:
        phi_end += 2 * math.pi
    x2, y2 = CX + R * math.cos(phi_end), CY + R * math.sin(phi_end)
    ring.append(f'<path d="M{x1:.3f} {y1:.3f} A{R} {R} 0 0 1 {x2:.3f} {y2:.3f}" fill="none" stroke="{T["muted"]}" stroke-width="{STROKE_RING}" marker-end="url(#arrow)"/>')

hub_rect = (CX - HW / 2, CY - HH / 2, HW, HH)


def box_distance(u, hw, hh):
    ds = []
    if abs(u[0]) > 1e-9:
        ds.append(hw / abs(u[0]))
    if abs(u[1]) > 1e-9:
        ds.append(hh / abs(u[1]))
    return min(ds)


for s in stations:
    u = s["u"]
    ds = box_distance(u, SW / 2, SH / 2)
    dh = box_distance(u, HW / 2, HH / 2)
    p = s["c"]
    x1, y1 = p[0] - ds * u[0], p[1] - ds * u[1]
    x2, y2 = CX + (dh + GAP) * u[0], CY + (dh + GAP) * u[1]
    spokes.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{T["soft"]}" stroke-width="1.2" stroke-dasharray="5,4" marker-end="url(#arrow-soft)"/>')
    s["spoke_len"] = math.hypot(x2 - x1, y2 - y1)

# ------------------------------------------------------------------ nodes
nodes = []
for s in stations:
    x, y, w, h = s["rect"]
    focal = s.get("focal")
    nodes.append(box(x, y, w, h, T["accent_tint"] if focal else T["paper"], T["accent"] if focal else T["ink"], 1.4 if focal else 1))
    tw = len(s["tag"]) * 12 + 12
    tag_stroke = T["accent"] if focal else T["muted"]
    nodes.append(f'<rect x="{r1(x + 12)}" y="{r1(y + 8)}" width="{tw}" height="16" rx="2" fill="none" stroke="{tag_stroke}" stroke-width="0.8" stroke-opacity="0.6"/>')
    nodes.append(text(x + 12 + tw / 2, y + 20, s["tag"], 12, 500, T["accent"] if focal else T["muted"], "middle"))
    nodes.append(text(x + 12, y + 46, f'{s["zh"]} {s["en"]}', 16, 600))
    nodes.append(text(x + 12, y + 66, s["sub"], 12, 400, T["muted"], family=MONO))
hx, hy, hw_, hh_ = hub_rect
nodes.append(box(hx, hy, hw_, hh_, T["ink"], T["ink"], 1))
nodes.append(text(CX, CY - 4, "狀態 State", 16, 600, T["paper"], "middle"))
nodes.append(text(CX, CY + 18, "θ, L, g", 12, 400, T["paper"], "middle", MONO, extra='fill-opacity="0.8"'))

# ------------------------------------------------------------------ side columns (plain sentences + English gloss)
def col(x, y, lines, size=14, fill=None, lh=22, weight=400):
    return "".join(text(x, y + i * lh, ln, size, weight, fill or T["ink"]) for i, ln in enumerate(lines))


left = "".join([
    text(40, 160, "一圈只做一件事", 16, 600), text(40, 180, "One pass does one thing", 12, 400, T["muted"]),
    col(40, 212, ["用目前的參數 θ 預測，", "量出錯得多嚴重，找出", "往哪走會更好，走一小步，", "再把新的 θ 帶回下一圈。"]),
    col(40, 312, ["Predict with the current θ, measure how", "wrong it is, find a better direction, take", "a small step, and carry the new θ into", "the next pass."], 12, T["muted"], 17),
    line(40, 400, 320, 400, T["rule"], 1),
    text(40, 428, "三個角色", 16, 600), text(40, 448, "Three roles", 12, 400, T["muted"]),
    text(40, 480, "假設類別　模型能表示什麼", 14), text(40, 498, "Hypothesis class: what it can represent", 12, 400, T["muted"]),
    text(40, 530, "損失　替每組參數打分", 14), text(40, 548, "Loss: scores every choice of parameters", 12, 400, T["muted"]),
    text(40, 580, "優化器　決定怎麼走", 14), text(40, 598, "Optimizer: decides how to move", 12, 400, T["muted"]),
])
update_formula = text(960, 212, fx(["θ ← θ − η ∇L(θ)"], 14), 14, 500, T["ink"], family=MONO, raw=True)
right = "".join([
    text(960, 160, "一次更新", 16, 600), text(960, 180, "One update step", 12, 400, T["muted"]),
    update_formula,
    col(960, 244, ["沿損失下降最快的方向", "走一步，步長 η 叫", "學習率。"]),
    col(960, 312, ["Step along the steepest descent", "direction; η is the learning rate."], 12, T["muted"], 17),
    line(960, 360, 1240, 360, T["rule"], 1),
    text(960, 388, "何時停？", 16, 600), text(960, 408, "When to stop?", 12, 400, T["muted"]),
    col(960, 440, ["達到收斂準則就停", "（KP3），否則再繞一圈。"]),
    col(960, 488, ["Stop when a convergence criterion", "holds (KP3); otherwise loop again."], 12, T["muted"], 17),
])

body = "".join([
    header("訓練就是反覆搜尋參數：預測、打分、求梯度、更新",
           "Training is a repeated parameter search: predict, score, differentiate, update"),
    "".join(ring), "".join(spokes), "".join(nodes), left, right,
    footer("實線＝一圈的順序；虛線＝每站把結果寫進中央的「狀態」；標籤＝該站屬於哪個角色。",
           "Solid arrows: order of one pass. Dashed: each station writes to the shared state. Tag: the station's role.",
           "只用目前位置的斜率，看不到別的山谷；可能停在局部極小或鞍點。",
           "It sees only the local slope, so it may stop at a local minimum or a saddle point."),
])
html_text = page(SLUG, "訓練是反覆的參數搜尋 Training as parameter search",
                 "Loop diagram: predict with parameters, compute the loss, take the gradient, choose a step and update the parameters, writing every result into a shared training state.",
                 body)

if __name__ == "__main__":
    print("spoke lengths (px):", [round(s["spoke_len"]) for s in stations])
    print("station rects:", [tuple(round(v) for v in s["rect"]) for s in stations])
    write_outputs(SLUG, html_text)
