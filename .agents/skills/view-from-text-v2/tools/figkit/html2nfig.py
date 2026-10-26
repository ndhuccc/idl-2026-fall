"""html2nfig.py -- turn a designed HTML page into NFig primitives (native JianBauDau nodes + a light-theme lecture PNG).

Design the page in HTML/CSS/inline SVG at the exact figure size (e.g. 1160 x 470) with the dark deck theme, fonts >= 16 px, flat or gradient fills
(a gradient is flattened to its middle colour; shadows are dropped). Math: put `<span class="k" data-t="LaTeX"></span>`; the page renders it with KaTeX
and the extractor emits a `$...$` math text node. Colours near #f2ede4 become the theme key TXT, near #b7b0a5 MUT, exactly #1c1712 BG.

    prims = extract("page.html", 1160, 470)        # list of ("rect"|"ellipse"|"line"|"text", dict)
    fig = to_nfig(prims, "fig-id", 1160, 470, alt)  # an NFig ready for the deck (freeform) and to_png
"""
import json
from .parts import NFig
from nativefig import _text_w as _tw

JS = r"""
() => {
  const out = [];
  const rgba = (s) => { const m = s.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(',').map(x => parseFloat(x)); return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1}; };
  const hex = (c) => '#' + [c.r, c.g, c.b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
  const col = (s, op = 1) => { const c = rgba(s); if (!c) return null; const a = c.a * op; if (a < 0.02) return null; return a >= 0.98 ? hex(c) : `rgba(${Math.round(c.r)},${Math.round(c.g)},${Math.round(c.b)},${a.toFixed(3)})`; };
  const gradMid = (bi) => { const m = bi.match(/rgba?\([^)]+\)/g); if (!m) return null; const c = m.map(rgba); const k = Math.floor(c.length / 2); return {r: (c[0].r + c[c.length - 1].r) / 2, g: (c[0].g + c[c.length - 1].g) / 2, b: (c[0].b + c[c.length - 1].b) / 2, a: 1}; };
  const R = (e) => e.getBoundingClientRect();
  const walk = (el) => {
    const cs = getComputedStyle(el); if (cs.display === 'none' || cs.visibility === 'hidden') return;
    const tag = el.tagName.toLowerCase();
    if (tag === 'svg') { svg(el); return; }
    if (['script', 'style', 'link', 'head'].includes(tag)) return;
    const r = R(el); const op = parseFloat(cs.opacity);
    let fill = col(cs.backgroundColor, op); if (!fill && cs.backgroundImage && cs.backgroundImage.includes('gradient')) { const g = gradMid(cs.backgroundImage); if (g) fill = hex(g); }
    const bw = parseFloat(cs.borderTopWidth) || 0; const stroke = bw > 0 && cs.borderTopStyle !== 'none' ? col(cs.borderTopColor, op) : null;
    const isK = el.classList.contains('k');
    const kids = [...el.childNodes]; const hasElemKid = kids.some(n => n.nodeType === 1 && !['BR', 'SUB', 'SUP', 'B', 'I', 'SPAN'].includes(n.tagName) );
    const txt = isK ? null : kids.filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim();
    const inlineOnly = !isK && !kids.some(n => n.nodeType === 1 && !['BR', 'SUB', 'SUP', 'B', 'I'].includes(n.tagName));
    const radius = Math.min(parseFloat(cs.borderTopLeftRadius) || 0, Math.min(r.width, r.height) / 2);
    const hasText = (isK || (inlineOnly && el.innerText && el.innerText.trim().length > 0));
    let chipText = null;
    if (hasText && !isK && fill && !stroke) { const rg0 = document.createRange(); rg0.selectNodeContents(el); const rs0 = [...rg0.getClientRects()].filter(q => q.width > 0); const fs0 = parseFloat(cs.fontSize);
      if (rs0.length && Math.max(...rs0.map(q => q.bottom)) - Math.min(...rs0.map(q => q.top)) < fs0 * 1.6) chipText = {s: el.innerText.trim(), size: fs0, bold: parseInt(cs.fontWeight) >= 600, color: col(cs.color, op) || '#f2ede4'}; }
    if (r.width > 0 && r.height > 0 && (fill || stroke) && !(hasText && isK)) out.push({k: 'rect', x: r.left, y: r.top, w: r.width, h: r.height, fill, stroke, sw: stroke ? bw : 0, radius, dash: false, hasText: !!chipText, chip: chipText});
    if (chipText) return;
    if (hasText) {
      let s, math = false, rr;
      if (isK) { s = '$' + el.dataset.t + '$'; math = true; rr = r; }
      else { s = el.innerText.replace(/\n\s*/g, '\n').trim(); const rg = document.createRange(); rg.selectNodeContents(el); const rs = [...rg.getClientRects()].filter(q => q.width > 0); rr = rs.length ? {left: Math.min(...rs.map(q => q.left)), right: Math.max(...rs.map(q => q.right)), top: Math.min(...rs.map(q => q.top)), bottom: Math.max(...rs.map(q => q.bottom)), width: 0, height: 0} : r; rr.width = rr.right - rr.left; rr.height = rr.bottom - rr.top; }
      const ta = cs.textAlign; const ha = ta === 'center' ? 'center' : (ta === 'right' || ta === 'end' ? 'right' : 'left');
      const x = ha === 'center' ? (rr.left + rr.right) / 2 : (ha === 'left' ? rr.left : rr.right);
      out.push({k: 'text', x, y: (rr.top + rr.bottom) / 2, s, size: parseFloat(cs.fontSize), bold: parseInt(cs.fontWeight) >= 600, ha, color: col(cs.color, op) || '#f2ede4', math, w: Math.round(Math.max(rr.width, r.width)) + (math ? 20 : 4), name: el.dataset.name || null, h: math ? Math.round(R(el.querySelector('.katex') || el).height) + 10 : null});
      return;
    }
    for (const c of el.children) walk(c);
  };
  const svg = (s) => {
    const sr = R(s); const vb = s.viewBox.baseVal; const sx = vb && vb.width ? sr.width / vb.width : 1, sy = vb && vb.height ? sr.height / vb.height : 1;
    const X = (v) => sr.left + v * sx, Y = (v) => sr.top + v * sy;
    for (const e of s.querySelectorAll('*')) {
      const cs = getComputedStyle(e); const tag = e.tagName.toLowerCase(); const op = parseFloat(e.getAttribute('opacity') || 1);
      const fa = e.getAttribute('fill'), sa = e.getAttribute('stroke'); const sw = parseFloat(e.getAttribute('stroke-width') || 0) * sx;
      const f = fa && fa !== 'none' && !fa.startsWith('url') ? col(fa.startsWith('#') ? hexToRgb(fa) : fa, op) : null; const st = sa && sa !== 'none' ? col(sa.startsWith('#') ? hexToRgb(sa) : sa, op) : null;
      if (tag === 'circle') { const cx = X(+e.getAttribute('cx')), cy = Y(+e.getAttribute('cy')), r = +e.getAttribute('r') * sx; const url = fa && fa.startsWith('url');
        const ring = e.dataset.ring === '1';
        if (st && sw > 0 && (f || url)) { out.push({k: 'rect', x: cx - r - sw / 2, y: cy - r - sw / 2, w: 2 * r + sw, h: 2 * r + sw, fill: f || '#2b2620', stroke: st, sw, radius: r + sw / 2, dash: false, hasText: false}); }
        else if (st && sw > 0) out.push({k: 'rect', x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: null, stroke: st, sw, radius: r, dash: false, hasText: false});
        else if (f) out.push({k: 'ellipse', cx, cy, d: 2 * r, fill: f}); }
      else if (tag === 'line') { if (st) out.push({k: 'line', x1: X(+e.getAttribute('x1')), y1: Y(+e.getAttribute('y1')), x2: X(+e.getAttribute('x2')), y2: Y(+e.getAttribute('y2')), color: st, lw: sw || 1}); }
      else if (tag === 'rect') { const x = X(+e.getAttribute('x')), y = Y(+e.getAttribute('y')); out.push({k: 'rect', x, y, w: +e.getAttribute('width') * sx, h: +e.getAttribute('height') * sy, fill: f, stroke: st, sw: st ? sw : 0, radius: +(e.getAttribute('rx') || 0) * sx, dash: false, hasText: false}); }
      else if (tag === 'polygon') { const b = R(e); if (f) out.push({k: 'text', x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 + 2, s: '★', size: Math.round(b.width * 1.15), bold: true, ha: 'center', color: f, math: false, w: Math.round(b.width * 1.6), name: null}); }
      else if (tag === 'text') { const b = R(e); const anc = e.getAttribute('text-anchor') || 'start'; const ha = anc === 'middle' ? 'center' : (anc === 'end' ? 'right' : 'left');
        out.push({k: 'text', x: ha === 'center' ? (b.left + b.right) / 2 : (ha === 'left' ? b.left : b.right), y: (b.top + b.bottom) / 2, s: e.textContent, size: parseFloat(e.getAttribute('font-size') || 16) * sx, bold: parseInt(e.getAttribute('font-weight') || 400) >= 600, ha, color: col(fa && fa.startsWith('#') ? hexToRgb(fa) : (fa || cs.fill), op) || '#f2ede4', math: false, w: Math.round(b.width) + 8, name: null}); }
    }
  };
  const hexToRgb = (h) => { h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(c => c + c).join(''); return `rgb(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)})`; };
  walk(document.body);
  return out;
}
"""

THEME_KEYS = {"#f2ede4": "TXT", "#b7b0a5": "MUT", "#1c1712": "BG"}


def extract(path, w, h):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": w, "height": h}); pg.goto("file://" + path); pg.wait_for_timeout(900)
        prims = pg.evaluate(JS); b.close()
    return prims


def screenshot(path, png, w, h):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": w, "height": h}); pg.goto("file://" + path); pg.wait_for_timeout(900); pg.screenshot(path=png); b.close()


def _c(c, text=False):
    if text and c == "#1c1712": return c
    return THEME_KEYS.get(c, c) if isinstance(c, str) else c


def to_nfig(prims, fid, w, h, alt=""):
    f = NFig(fid, w, h, alt)
    for d in prims:
        k = d["k"]
        if k == "rect":
            ch = d.get("chip")
            if ch: f.rect(d["x"], d["y"], d["w"], d["h"], fill=_c(d["fill"]), stroke=_c(d["stroke"]), sw=d["sw"], radius=d["radius"], text=ch["s"], tcolor=ch["color"] if ch["color"] != "#f2ede4" else "TXT", size=round(ch["size"]), bold=ch["bold"])
            else: f.rect(d["x"], d["y"], d["w"], d["h"], fill=_c(d["fill"]), stroke=_c(d["stroke"]), sw=d["sw"], radius=d["radius"])
        elif k == "ellipse": f.ellipse(d["cx"], d["cy"], d["d"], _c(d["fill"]))
        elif k == "line": f.line(d["x1"], d["y1"], d["x2"], d["y2"], _c(d["color"]), d["lw"])
        elif k == "text": f.text(d["x"], d["y"] + (6 if d["math"] and "\\left" in d["s"] else 0), d["s"], _c(d["color"], True), max(16, round(d["size"])) if d["size"] < 16 else round(d["size"]), d["bold"], d["ha"], math=d["math"], w=(d["w"] if d["math"] else round(max(d["w"], _tw(d["s"], d["size"], d["bold"]) / 1.15 + 14))), name=d["name"], h=(max(d.get("h") or 0, round(d["size"] * 2.75)) if d["math"] and "\\left" in d["s"] else d.get("h")))
    return f


def save(prims, path): json.dump(prims, open(path, "w"), ensure_ascii=False)
def load(path): return json.load(open(path))
