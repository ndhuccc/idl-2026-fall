"""KP04 design helpers on top of view-from-text-v3 htmlkit.
New here: mt() = one text node that mixes words and inline LaTeX ($...$). In the HTML design the math is rendered by KaTeX
(so lint and screenshots measure the real width); on conversion the node becomes ONE native text node whose text keeps the
$...$ source (JianBauDau renders inline KaTeX), instead of htmlkit's math-only k() nodes. Never put math inside chip()."""
import os, sys, html as _html, json
SK = "/workspace/idl-2026-fall/.agents/skills"
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text/tools/deckkit"): sys.path.insert(0, os.path.join(SK, p))
os.environ.setdefault("KATEX_DIR", os.path.join(SK, "slides-to-html/node_modules/katex/dist"))
import htmlkit as HK
from htmlkit import card, head, chip, txt, named, svg, rgb, rgba, hexc, ROLE, INK, TXT, MUT, BG, _p, page, lint
from figkit import html2nfig as H

W, HGT = 1160, 460
def _ast(tex): return tex.replace("*", r"\ast ")      # a bare * inside $...$ could be read as Markdown emphasis by the player
def k(x, y, w, tex, size=22, h=40): return HK.k(x, y, w, _ast(tex), size, h)
OR, BL, GR, RD, YE, VI, TE, GY = (hexc(c) for c in ("orange", "blue", "green", "red", "yellow", "violet", "teal", "grey"))
def tr(lang): return lambda zh, en: zh if lang == "zh" else en

def _mt_inner(s):
    out = []
    for ln in s.split("\n"):
        parts = ln.split("$"); seg = []
        for i, part in enumerate(parts):
            if i % 2: seg.append(f'<span class="kin" data-t="{_html.escape(_ast(part), quote=True)}"></span>')
            else: seg.append(_html.escape(part))
        out.append("".join(seg))
    return "<br>".join(out)

def mt(x, y, w, s, size=18, color=TXT, bold=False, ha="left", lh=None, name=None, h=None):
    """Mixed words + $inline math$ (one native text node). '\\n' breaks lines; no automatic wrapping."""
    s = "$".join(_ast(p) if i % 2 else p for i, p in enumerate(s.split("$")))
    n = s.count("\n") + 1; lh = lh or round(size * 1.5); hh = h or n * lh
    nm = f' data-name="{_html.escape(name, quote=True)}"' if name else ""
    return (f'<div data-wrap="1" data-mt="{_html.escape(s, quote=True)}"{nm} style="{_p(x, y, w, hh)}font-size:{size}px;color:{color};font-weight:{700 if bold else 400};'
            f'text-align:{ha};line-height:{lh}px;white-space:nowrap;padding-top:{max(0, (hh - n * lh) // 2)}px">{_mt_inner(s)}</div>')

def box(x, y, w, h, fill=None, stroke=None, sw=2, r=10):
    st = f"background:{fill};" if fill else ""
    if stroke: st += f"border:{sw}px solid {stroke};"
    return f'<div style="{_p(x, y, w, h)}{st}border-radius:{r}px"></div>'

def mchip(x, y, w, role, s, solid=False, h=34, size=18, bold=True):
    """A chip whose label may contain $math$. Without math it is htmlkit's chip (one node); with math a tinted rect + light mt label
    (a dark label on a separate rect would fail the E10 contrast check, which compares text with the canvas)."""
    if "$" not in s: return chip(x, y, w, role, s, solid, h, size)
    solid = False
    return box(x, y, w, h, rgb(role) if solid else rgba(role, .28), None, 0, 10) + mt(x, y, w, s, size, INK if solid else TXT, bold, "center", lh=h, h=h)

def cell(x, y, v, role="grey", w=60, h=52, size=24, a=.30, dim=False, bold=True, stroke_a=.9, tcolor=TXT):
    """A number cell: tinted fill, outline, centred value (v is put in $...$ so minus signs are typeset)."""
    if dim: a, stroke_a = .08, .35
    lab = "" if v is None or v == "" else mt(x, y, w, f"${v}$", size, MUT if dim else tcolor, bold, "center", lh=h, h=h)
    return box(x, y, w, h, rgba(role, a), rgba(role, stroke_a), 2, 8) + lab

def arrow(x1, y1, x2, y2, color=MUT, lw=3, hs=11):
    import math
    a = math.atan2(y2 - y1, x2 - x1); bx, by = x2 - hs * math.cos(a), y2 - hs * math.sin(a)
    px, py = -math.sin(a) * hs * .6, math.cos(a) * hs * .6
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{lw}"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {bx+px:.1f},{by+py:.1f} {bx-px:.1f},{by-py:.1f}" fill="{color}"/>')

def take(t, zh, en, y=410, role="yellow"):
    return box(0, y, W, 48, rgba(role, .14), rgba(role, .55), 2, 12) + mt(16, y + 7, W - 32, t(zh, en), 19, TXT, True, "center", lh=34, h=34)

def added(x, y, lang, w=None, role="violet"):
    """The small chip that marks content the lecture tags [補充] / [Added]."""
    s = "補充" if lang == "zh" else "Added"; w = w or (56 if lang == "zh" else 70)
    return chip(x, y, w, role, s, True, 26, 16)

KIN_JS = '<script>document.querySelectorAll(".kin").forEach(e=>katex.render(e.dataset.t,e,{throwOnError:false}));</script>'

MEASURE_JS = r"""
() => { const out = [];
  const hex = (s) => { const m = s.match(/rgba?\(([^)]+)\)/); const p = m[1].split(',').map(parseFloat); return '#' + p.slice(0, 3).map(v => Math.round(v).toString(16).padStart(2, '0')).join(''); };
  for (const e of [...document.querySelectorAll('[data-mt]')]) {
    const rg = document.createRange(); rg.selectNodeContents(e); const rs = [...rg.getClientRects()].filter(q => q.width > 0 && q.height > 0);
    const cs = getComputedStyle(e); const b = e.getBoundingClientRect();
    const l = Math.min(...rs.map(q => q.left)), r = Math.max(...rs.map(q => q.right)), t = Math.min(...rs.map(q => q.top)), bt = Math.max(...rs.map(q => q.bottom));
    const katexErr = e.querySelectorAll('.katex-error').length;
    out.push({src: e.dataset.mt, l, r, t, b: bt, bl: b.left, br: b.right, bt: b.top, bb: b.bottom, size: parseFloat(cs.fontSize), bold: parseInt(cs.fontWeight) >= 600,
              ha: cs.textAlign, color: hex(cs.color), name: e.dataset.name || null, err: katexErr});
    e.remove(); }
  return out; }
"""

def build(fid, design, outdir, langs=("zh", "en"), w=W, h=HGT):
    """Like htmlkit.build, plus mt() nodes. Writes <fid>_<lang>.html/.png/.json; prints lint + KaTeX errors."""
    from playwright.sync_api import sync_playwright
    outdir = os.path.abspath(outdir); os.makedirs(outdir, exist_ok=True); report = {}
    for lg in langs:
        p = os.path.join(outdir, f"{fid}_{lg}.html"); open(p, "w").write(page(design(lg) + KIN_JS, w, h))
        H.screenshot(p, os.path.join(outdir, f"{fid}_{lg}.png"), w, h)
        warns = lint(p, w, h)
        with sync_playwright() as pw:
            b = pw.chromium.launch(); pg = b.new_page(viewport={"width": w, "height": h}); pg.goto("file://" + p); pg.wait_for_timeout(900)
            errs = pg.evaluate("() => [...document.querySelectorAll('.katex-error')].map(e => e.textContent).slice(0, 5)")
            mts = pg.evaluate(MEASURE_JS); prims = pg.evaluate(H.JS); b.close()
        for m in mts:
            if m["l"] < -1 or m["r"] > w + 1 or m["bt"] < -1 or m["bb"] > h + 1: warns.append("mt outside the canvas: " + m["src"][:30])
            if m["r"] > m["br"] + 2 or m["l"] < m["bl"] - 2: warns.append("mt wider than its box: " + m["src"][:30])
            ha = "center" if m["ha"] == "center" else ("right" if m["ha"] in ("right", "end") else "left")
            x = (m["l"] + m["r"]) / 2 if ha == "center" else (m["l"] if ha == "left" else m["r"])
            nl = m["src"].count("\n") + 1
            prims.append({"k": "text", "x": x, "y": (m["bt"] + m["bb"]) / 2, "s": m["src"], "size": m["size"], "bold": m["bold"], "ha": ha, "color": m["color"],
                          "math": True, "w": round(m["r"] - m["l"]) + 20, "name": m["name"], "h": round(m["bb"] - m["bt"]) if nl > 1 else None})
        if errs: warns.append("KaTeX errors: " + " | ".join(errs))
        H.save(prims, os.path.join(outdir, f"{fid}_{lg}.json")); report[lg] = warns
        print(f"[{fid} {lg}] {len(prims)} primitives" + ("" if not warns else "; lint: " + " | ".join(warns)))
    return report

def native(fid, figid, jsondir, alt, lang=None, w=W, h=HGT):
    lang = lang or ("en" if os.environ.get("KP_LANG", "zh") == "en" else "zh")
    return H.to_nfig(H.load(os.path.join(jsondir, f"{fid}_{lang}.json")), figid, w, h, alt)

def mhead(x, y, w, role, s, h=36, size=18):
    """Card header that may contain $math$: tinted bar + light bold mt (head() would need dark text on a separate rect)."""
    return box(x, y, w, h, rgba(role, .32), None, 0, 12) + mt(x, y, w, s, size, TXT, True, "center", lh=h, h=h)
