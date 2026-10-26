"""htmlkit.py -- design a figure as HTML/CSS/SVG at the exact figure size, then turn it into native JianBauDau nodes (view-from-text-v3).

Typical use (see examples/kp18_pnn.py):

    from htmlkit import *
    def design(lang):                    # returns the BODY html of one figure; strings come from a {"zh":..,"en":..} dict
        return card(0,0,330,400,"blue") + head(0,0,330,"blue","① ...") + ...
    build("kp18_pnn", design, W, H, outdir)   # writes <fid>_<lang>.html/.png/.json and prints lint warnings
    fig = native("kp18_pnn", "lec05-kp18-arch", W, H, jsondir, alt)   # NFig (freeform page / lecture PNG)

Rules baked in: dark deck theme, text >= 16 px, the 8-role palette of figkit.parts, flat fills (a gradient is flattened to its middle colour), math via
`k(...)` (KaTeX in the page, `$...$` native math node), explicit <br> for line breaks (the player wraps by its own width estimate).
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", "..", "view-from-text-v2", "tools"))
sys.path.insert(0, V2)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "view-from-text", "tools", "deckkit")))
from figkit import html2nfig as H

KATEX = os.environ.get("KATEX_DIR", os.path.abspath(os.path.join(HERE, "..", "..", "slides-to-html", "node_modules", "katex", "dist")))
ROLE = {"red": (224, 122, 112), "blue": (106, 159, 214), "green": (125, 187, 140), "orange": (217, 119, 87), "violet": (169, 139, 208), "yellow": (224, 200, 95), "teal": (95, 184, 168), "grey": (183, 176, 165)}
INK, TXT, MUT, BG = "#1c1712", "#f2ede4", "#b7b0a5", "#1c1712"


def rgb(r): return "rgb(%d,%d,%d)" % ROLE[r]
def rgba(r, a): return "rgba(%d,%d,%d,%s)" % (*ROLE[r], a)
def hexc(r): return "#%02x%02x%02x" % ROLE[r]


def _p(x, y, w, h): return f"position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;"


def card(x, y, w, h, role, a=0.12):
    """A tinted panel with an outline (the usual card)."""
    return f'<div style="{_p(x, y, w, h)}background:{rgba(role, a)};border:2px solid {rgba(role, .65)};border-radius:14px"></div>'


def head(x, y, w, role, s, h=34, size=17):
    """A solid header chip (dark text on the bright role colour)."""
    return f'<div style="{_p(x, y, w, h)}background:{rgb(role)};border-radius:12px;color:{INK};font-size:{size}px;font-weight:700;text-align:center;line-height:{h}px">{s}</div>'


def chip(x, y, w, role, s, solid=False, h=30, size=16):
    """One-line rounded label. solid=True: bright fill + dark text (verdicts); else tinted fill + light text."""
    return f'<div style="{_p(x, y, w, h)}background:{rgb(role) if solid else rgba(role, .28)};border-radius:10px;color:{INK if solid else TXT};font-size:{size}px;font-weight:700;text-align:center;line-height:{h}px">{s}</div>'


def txt(x, y, w, s, size=16, color=TXT, bold=False, ha="center", h=24):
    """A text block (use <br> for line breaks; h = line height * number of lines)."""
    return f'<div style="{_p(x, y, w, h)}font-size:{size}px;color:{color};font-weight:{700 if bold else 400};text-align:{ha};line-height:{round(h / (s.count("<br>") + 1))}px">{s}</div>'


def k(x, y, w, tex, size=22, h=40):
    """A KaTeX formula (centred). `tex` is plain LaTeX with single backslashes."""
    return f'<div style="{_p(x, y, w, h)}font-size:{size}px;color:{TXT};text-align:center"><span class="k" data-t="{tex.replace(chr(34), "&quot;")}"></span></div>'


def bar(x, y, w, h, role, s, size=20):
    """A value bar (gradient between two shades of the role; flattened on conversion) with a centred label."""
    r0, g0, b0 = ROLE[role]
    return f'<div style="{_p(x, y, w, h)}background:linear-gradient(90deg,rgb({r0},{g0},{b0}),rgb({int(r0*.78)},{int(g0*.78)},{int(b0*.78)}));border-radius:10px;color:{INK};font-size:{size}px;font-weight:700;text-align:center;line-height:{h}px">{s}</div>'


def named(x, y, w, s, name, size=16, color=MUT, h=24, ha="left"):
    """A text block that is a term-definition node (name = 'term:<詞>') for the term audit."""
    return f'<div data-name="{name}" style="{_p(x, y, w, h)}font-size:{size}px;color:{color};text-align:{ha};line-height:{h}px">{s}</div>'


def svg(x, y, w, h, inner): return f'<svg style="position:absolute;left:{x}px;top:{y}px" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{inner}</svg>'


def page(body, w, h):
    return (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="file://{KATEX}/katex.min.css"><script src="file://{KATEX}/katex.min.js"></script>'
            f'<style>*{{box-sizing:border-box}}body{{margin:0;width:{w}px;height:{h}px;background:{BG};font-family:"Noto Sans CJK TC","WenQuanYi Zen Hei",sans-serif;position:relative}}</style></head><body>{body}'
            '<script>document.querySelectorAll(".k").forEach(e=>katex.render(e.dataset.t,e,{throwOnError:false}));</script></body></html>')


LINT_JS = r"""
([W, H]) => {
  const out = []; const rd = (s) => { const m = s.match(/rgba?\(([^)]+)\)/); return m ? m[1].split(',').map(parseFloat) : null; };
  for (const e of document.querySelectorAll('body *')) {
    if (e.closest('svg') && e.tagName.toLowerCase() !== 'svg') {
      if (e.tagName.toLowerCase() === 'text') { const fs = parseFloat(e.getAttribute('font-size') || 16) * (e.ownerSVGElement.getBoundingClientRect().width / (e.ownerSVGElement.viewBox.baseVal.width || 1)); if (fs < 15.9) out.push('svg text <16px: ' + e.textContent.slice(0, 20)); }
      continue; }
    const cs = getComputedStyle(e); const r = e.getBoundingClientRect();
    if (e.classList.contains('katex') || e.closest('.katex')) continue;
    const own = [...e.childNodes].filter(n => n.nodeType === 3 && n.textContent.trim()).map(n => n.textContent.trim()).join('');
    if (own && parseFloat(cs.fontSize) < 15.9) out.push('text <16px: ' + own.slice(0, 20));
    if (r.width > 0 && (r.left < -1 || r.top < -1 || r.right > W + 1 || r.bottom > H + 1)) out.push('outside the canvas: ' + (own || e.className || e.tagName).toString().slice(0, 24));
    if (own) { const rg = document.createRange(); rg.selectNodeContents(e); const rs = [...rg.getClientRects()].filter(q => q.width > 0); if (rs.length) { const right = Math.max(...rs.map(q => q.right)), bottom = Math.max(...rs.map(q => q.bottom)); if (right > r.right + 2 || bottom > r.bottom + 2) out.push('text overflows its box: ' + own.slice(0, 20)); } }
    if (own && !e.querySelector('br') && own.length > 3) { const rg = document.createRange(); rg.selectNodeContents(e); const rs = [...rg.getClientRects()].filter(q => q.width > 0); const tops = new Set(rs.map(q => Math.round(q.top))); if (tops.size > 1 && !e.dataset.wrap) out.push('wraps by itself (add <br>): ' + own.slice(0, 20)); }
  }
  return out;
}
"""


def lint(path, w, h):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": w, "height": h}); pg.goto("file://" + path); pg.wait_for_timeout(900); res = pg.evaluate(LINT_JS, [w, h]); b.close()
    return res


def build(fid, design, w, h, outdir, langs=("zh", "en")):
    outdir = os.path.abspath(outdir); os.makedirs(outdir, exist_ok=True)
    for lg in langs:
        p = os.path.join(outdir, f"{fid}_{lg}.html"); open(p, "w").write(page(design(lg), w, h))
        prims = H.extract(p, w, h); H.save(prims, os.path.join(outdir, f"{fid}_{lg}.json")); H.screenshot(p, os.path.join(outdir, f"{fid}_{lg}.png"), w, h)
        warns = lint(p, w, h); print(f"[{lg}] {len(prims)} primitives" + ("" if not warns else "; lint: " + " | ".join(warns)))


def native(fid, figid, w, h, jsondir, alt, lang=None):
    lang = lang or ("en" if os.environ.get("KP_LANG", "zh") == "en" else "zh")
    return H.to_nfig(H.load(os.path.join(jsondir, f"{fid}_{lang}.json")), figid, w, h, alt)
