"""KP18 page 3 designed in HTML (1160 x 470, dark theme, fonts >= 16 px) and converted to native nodes by tools/figkit/html2nfig.py."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools")); sys.path.insert(0, HERE)
from pathlib import Path
_SK = Path(os.environ.get("COURSE_SKILLS") or (Path(os.environ["COURSE_REPO"]) / ".agents/skills" if os.environ.get("COURSE_REPO") else Path(__file__).resolve().parents[2]))
KATEX = Path(os.environ.get("KATEX_DIR", _SK / "slides-to-html/node_modules/katex/dist")).resolve().as_uri()
ROLE = {"blue": (106, 159, 214), "orange": (217, 119, 87), "green": (125, 187, 140), "yellow": (224, 200, 95), "grey": (183, 176, 165), "violet": (169, 139, 208)}
rgb = lambda r: "rgb(%d,%d,%d)" % ROLE[r]; rgba = lambda r, a: "rgba(%d,%d,%d,%s)" % (*ROLE[r], a)
L = {
 "zh": dict(c1="① 單位圓上的向量", c1s="每個向量長度都是 1", lgA="A 類 20° 40° 60°", lgB="B 類 100° 120°", test="測試 70°", pnnn="term:機率神經網路", pnn="機率神經網路（PNN）：把 Parzen 判別式搬進網路，幾乎不用訓練。", c2="② 一個訓練點一個神經元", inp="輸入層", hid="隱藏層：權重＝訓練向量", outl="輸出層", c3="③ 輸出與判定", A="A 類", B="B 類", win="A 勝 ★", note="70° 離 A 類的 60°、40°<br>較近，所以 A 的分數較高。", hl="隱藏層", ol="輸出層", mx="取最大者"),
 "en": dict(c1="① Vectors on the unit circle", c1s="every vector has length 1", lgA="A: 20° 40° 60°", lgB="B: 100° 120°", test="test 70°", pnnn="term:Probabilistic neural network", pnn="Probabilistic neural network (PNN): Parzen rule as a network.", c2="② One neuron per training point", inp="input", hid="hidden: weights = training vectors", outl="output", c3="③ Outputs and decision", A="class A", B="class B", win="A wins ★", note="70° is near class A's 60°<br>and 40°, so A scores higher.", hl="hidden", ol="output", mx="take the largest"),
}


def design(lang):
    t = L[lang]; angs = [20, 40, 60, 100, 120]; th = np.deg2rad(angs); dots = np.cos(th - np.deg2rad(70)); ys = np.exp((dots - 1) / 0.25); oA = 0.5 / 3 * ys[:3].sum(); oB = 0.5 / 2 * ys[3:].sum()
    cx, cy, R = 165, 150, 92
    def pt(a, r=R): q = math.radians(a); return cx + r * math.cos(q), cy - r * math.sin(q)
    s1 = f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#b7b0a5" stroke-width="2"/><line x1="{cx-R-10}" y1="{cy}" x2="{cx+R+10}" y2="{cy}" stroke="#6b645a" stroke-width="1.5"/><line x1="{cx}" y1="{cy+R+10}" x2="{cx}" y2="{cy-R-10}" stroke="#6b645a" stroke-width="1.5"/>'
    tx, ty = pt(70)
    for a in angs:
        x, y = pt(a); c = "#6a9fd6" if a < 90 else "#d97757"; s1 += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2" opacity="0.5"/>'
    s1 += f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#7dbb8c" stroke-width="4"/>'
    for a in angs:
        x, y = pt(a); c = "#6a9fd6" if a < 90 else "#d97757"; lx, ly = pt(a, R + 26); s1 += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{c}"/><text x="{lx:.1f}" y="{ly+6:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="{c}">{a}°</text>'
    s1 += f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="13" fill="#7dbb8c" stroke="#f2ede4" stroke-width="3"/>'
    s1 += f'<text x="6" y="20" font-size="16" font-weight="700" fill="#7dbb8c">★ {t["test"]}</text>'
    xi, xh = 36, 190; yin = [125, 235]; yh = [40 + k * 60 for k in range(5)]; ports = [(488, 100), (488, 240)]
    s2 = ""
    for a in yin:
        for b in yh: s2 += f'<line x1="{xi+22}" y1="{a}" x2="{xh-20}" y2="{b}" stroke="#6b645a" stroke-width="1.5"/>'
    for k, b in enumerate(yh):
        c = "#6a9fd6" if k < 3 else "#d97757"; p = ports[0] if k < 3 else ports[1]; s2 += f'<line x1="{xh+20}" y1="{b}" x2="{p[0]-10}" y2="{p[1]}" stroke="{c}" stroke-width="{2+ys[k]*5:.1f}" opacity="0.8"/>'
    for lab, a in (("x₁", yin[0]), ("x₂", yin[1])): s2 += f'<circle cx="{xi}" cy="{a}" r="22" fill="rgba(183,176,165,0.35)" stroke="#b7b0a5" stroke-width="3"/><text x="{xi}" y="{a+6}" text-anchor="middle" font-size="18" font-weight="700" fill="#f2ede4">{lab}</text>'
    for k, b in enumerate(yh):
        c = "#6a9fd6" if k < 3 else "#d97757"
        s2 += f'<rect x="{xh+30}" y="{b-23}" width="190" height="46" rx="8" fill="#1c1712"/><circle cx="{xh}" cy="{b}" r="20" fill="{"rgba(106,159,214,0.35)" if k < 3 else "rgba(217,119,87,0.35)"}" stroke="{c}" stroke-width="3"/><text x="{xh}" y="{b+6}" text-anchor="middle" font-size="16" font-weight="700" fill="#f2ede4">y{k+1}</text><text x="{xh-28}" y="{b+6}" text-anchor="end" font-size="16" fill="#b7b0a5">{angs[k]}°</text><text x="{xh+38}" y="{b-3}" font-size="16" fill="#b7b0a5">cos = {dots[k]:.4f}</text><text x="{xh+38}" y="{b+17}" font-size="16" font-weight="700" fill="{c}">y = {ys[k]:.4f}</text>'
    for p, c in zip(ports, ("#6a9fd6", "#d97757")): s2 += f'<circle cx="{p[0]}" cy="{p[1]}" r="10" fill="{c}"/>'
    def card(x, y, w, h, role): return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:{rgba(role,.12)};border:2px solid {rgba(role,.65)};border-radius:14px"></div>'
    def head(x, y, w, role, s): return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:34px;background:{rgb(role)};border-radius:12px;color:#1c1712;font-size:17px;font-weight:700;text-align:center;line-height:34px">{s}</div>'
    def txt(x, y, w, s, size=16, color="#f2ede4", bold=False, ha="center", h=24): return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;font-size:{size}px;color:{color};font-weight:{700 if bold else 400};text-align:{ha};line-height:{h}px">{s}</div>'
    def k(x, y, w, tex, size=22, h=40): return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;font-size:{size}px;color:#f2ede4;text-align:center"><span class="k" data-t="{tex}"></span></div>'
    def chip(x, y, w, role, s, solid=False): return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:30px;background:{rgb(role) if solid else rgba(role,.28)};border-radius:10px;color:{"#1c1712" if solid else "#f2ede4"};font-size:16px;font-weight:700;text-align:center;line-height:30px">{s}</div>'
    bw = lambda o: max(int(o / 0.3 * 236), 20)
    body = (card(0, 0, 330, 400, "blue") + head(0, 0, 330, "blue", t["c1"]) + txt(0, 40, 330, t["c1s"], 16, "#b7b0a5") + f'<svg style="position:absolute;left:0;top:68px" width="330" height="270" viewBox="0 0 330 270">{s1}</svg>'
            + chip(10, 346, 150, "blue", t["lgA"]) + chip(170, 346, 150, "orange", t["lgB"])
            + card(345, 0, 520, 400, "orange") + head(345, 0, 520, "orange", t["c2"]) + f'<div data-name="{t["pnnn"]}" style="position:absolute;left:355px;top:38px;width:500px;height:24px;font-size:16px;color:#b7b0a5;line-height:24px;text-align:left">{t["pnn"]}</div>' + f'<svg style="position:absolute;left:345px;top:56px" width="520" height="320" viewBox="0 0 520 330">{s2}</svg>'
            + txt(345 + 6, 366, 70, t["inp"], 16, "#b7b0a5") + txt(345 + 120, 366, 250, t["hid"], 16, "#b7b0a5") + txt(345 + 430, 366, 80, t["outl"], 16, "#b7b0a5")
            + card(880, 0, 280, 400, "green") + head(880, 0, 280, "green", t["c3"])
            + txt(894, 48, 100, t["A"], 18, "#6a9fd6", True, "left") + chip(1040, 46, 110, "yellow", t["win"], True) + k(894, 76, 200, "\\frac{0.5}{3}\\times %.4f" % ys[:3].sum(), 20, 44)
            + f'<div style="position:absolute;left:894px;top:126px;width:{bw(oA)}px;height:40px;background:linear-gradient(90deg,#6a9fd6,#4a7fb6);border-radius:10px;color:#1c1712;font-size:22px;font-weight:700;text-align:center;line-height:40px">{oA:.4f}</div>'
            + txt(894, 190, 100, t["B"], 18, "#d97757", True, "left") + k(894, 218, 200, "\\frac{0.5}{2}\\times %.4f" % ys[3:].sum(), 20, 44)
            + f'<div style="position:absolute;left:894px;top:268px;width:{bw(oB)}px;height:40px;background:linear-gradient(90deg,#d97757,#b85a3a);border-radius:10px;color:#1c1712;font-size:22px;font-weight:700;text-align:center;line-height:40px">{oB:.4f}</div>'
            + f'<div style="position:absolute;left:894px;top:322px;width:252px;height:62px;font-size:16px;color:#f2ede4;line-height:22px">{t["note"]}</div>'
            + f'<div style="position:absolute;left:0;top:410px;width:1160px;height:56px;background:{rgba("yellow",.16)};border:2px solid {rgba("yellow",.55)};border-radius:12px"></div>'
            + txt(20, 425, 90, t["hl"], 18, "#f2ede4", True, "left", 28) + k(110, 414, 330, "y_k=\\exp\\left(\\frac{x_k^{T}x-1}{h^{2}}\\right)", 22, 48) + txt(450, 425, 40, "→", 24, "#f2ede4", True, "center", 28)
            + txt(500, 425, 90, t["ol"], 18, "#f2ede4", True, "left", 28) + k(590, 414, 330, "\\frac{P(\\omega_m)}{N_m}\\sum_{k\\in m}y_k", 22, 48) + txt(930, 425, 220, t["mx"], 18, "#f2ede4", True, "left", 28))
    return f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{KATEX}/katex.min.css"><script src="{KATEX}/katex.min.js"></script><style>body{{margin:0;width:1160px;height:470px;background:#1c1712;font-family:"WenQuanYi Zen Hei",sans-serif;position:relative}}</style></head><body>{body}<script>document.querySelectorAll(".k").forEach(e=>katex.render(e.dataset.t,e,{{throwOnError:false}}));</script></body></html>'


def build(lang, workdir):
    from figkit import html2nfig as H
    os.makedirs(workdir, exist_ok=True); p = os.path.join(workdir, f"pnn_{lang}.html"); open(p, "w").write(design(lang)); prims = H.extract(p, 1160, 470); H.save(prims, os.path.join(workdir, f"pnn_{lang}.json"))
    H.screenshot(p, os.path.join(workdir, f"pnn_{lang}.png"), 1160, 470); return prims


def pnn_native():
    from figkit import html2nfig as H
    from figkit.parts import T
    lang = os.environ.get("KP_LANG", "zh"); prims = H.load(os.path.join(HERE, "kp18_pnn_html_%s.json" % lang))
    return H.to_nfig(prims, "lec05-kp18-arch", 1160, 470, "PNN: three layers; test vector at 70°, class A 0.2943 beats class B 0.2062 (h=0.5, priors 0.5).")


if __name__ == "__main__":
    wd = sys.argv[1]
    for lg in ("zh", "en"):
        pr = build(lg, wd); print(lg, len(pr))
