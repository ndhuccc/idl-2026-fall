"""Shared helpers for the diagram-design figures of idl-2026-fall (project skin = the dark deck palette).

Tokens come from .agents/skills/diagram-design/references/style-guide.md (Dark column, customised 2026-10-13).
Every figure script in this folder that starts with "dd_" or is named idl-*.py imports this module.

Outputs of a figure script (slug = file name without extension):
    assets/figures/<slug>.html   self-contained page (inline SVG, Google Fonts link)
    assets/figures/<slug>.png    PNG at 2x of the viewBox (what the slides use)
    assets/figures/<slug>.svg    standalone SVG (fonts by name; for editing, not for <img>)
"""
import html
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(HERE)
SKILL = os.path.abspath(os.path.join(FIG_DIR, "..", "..", ".agents", "skills", "diagram-design"))

T = dict(
    paper="#151513", paper2="#1f1f1c", ink="#f2ede4", ink_strong="#111111", muted="#b7b0a5", soft="#a39d92",
    rule="rgba(242,237,228,0.14)", rule_solid="rgba(183,176,165,0.32)", accent="#d97757",
    accent_tint="rgba(217,119,87,0.12)", link="#6a9fd6",
)
SANS = "'Geist','Noto Sans TC','PingFang TC','Microsoft JhengHei',sans-serif"
MONO = "'Geist Mono',ui-monospace,monospace"
FONT_LINK = ("https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600"
             "&family=Noto+Serif:ital@0;1&family=Noto+Sans+KR:wght@400;500;600&family=Noto+Serif+KR:wght@400"
             "&family=Noto+Sans+TC:wght@400;500;600&family=Noto+Serif+TC:wght@400&display=swap")


def esc(s):
    return html.escape(s, quote=False)


def r1(v):
    """Round to 1 decimal for SVG coordinates, dropping a trailing .0."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def text(x, y, content, size=12, weight=400, fill=None, anchor="start", family=SANS, raw=False, extra=""):
    fill = fill or T["ink"]
    body = content if raw else esc(content)
    return (f'<text x="{r1(x)}" y="{r1(y)}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{(" " + extra) if extra else ""}>{body}</text>')


def fx(parts, base=14):
    """Formula runs -> tspans. parts: str | (str, 'sub'|'sup'). Tracks the running dy offset (primitive-math-notation.md §2)."""
    small = max(7, round(base * 0.75))
    out, off = [], 0
    for p in parts:
        txt, mode = (p, None) if isinstance(p, str) else p
        size, shift = (small, 3) if mode == "sub" else (small, -5) if mode == "sup" else (base, 0)
        dy = shift - off
        off = shift
        out.append(f'<tspan font-size="{size}"' + (f' dy="{dy}"' if dy else "") + f">{esc(txt)}</tspan>")
    if off:
        out.append(f'<tspan font-size="{base}" dy="{-off}"></tspan>')
    return "".join(out)


def box(x, y, w, h, fill=None, stroke=None, sw=1, rx=6, dash=None):
    """Opaque paper mask first, then the styled box (primitives-core.md § Node box)."""
    fill = fill or T["paper"]
    stroke = stroke or T["ink"]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{r1(x)}" y="{r1(y)}" width="{r1(w)}" height="{r1(h)}" rx="{rx}" fill="{T["paper"]}"/>'
            f'<rect x="{r1(x)}" y="{r1(y)}" width="{r1(w)}" height="{r1(h)}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=None, sw=1, dash=None, marker=None):
    stroke = stroke or T["muted"]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{r1(x1)}" y1="{r1(y1)}" x2="{r1(x2)}" y2="{r1(y2)}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def defs():
    def mk(i, c):
        return f'<marker id="{i}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{c}"/></marker>'
    return "<defs>" + mk("arrow", T["muted"]) + mk("arrow-accent", T["accent"]) + mk("arrow-link", T["link"]) + mk("arrow-soft", T["soft"]) + "</defs>"


def header(title, english, mnemonic=None):
    parts = [text(40, 84, title, 40, 600),
             text(40, 112, english, 16, 500, T["muted"])]
    if mnemonic:
        parts.append(text(1240, 84, mnemonic, 16, 500, T["muted"], "end"))
    parts.append(line(40, 128, 1240, 128, T["rule_solid"], 1))
    return "".join(parts)


def footer(key_zh, key_en, limit_zh, limit_en):
    return "".join([
        line(40, 660, 1240, 660, T["rule"], 1),
        text(40, 684, key_zh, 12, 400, T["muted"]),
        text(40, 702, key_en, 11, 400, T["soft"]),
        text(1240, 684, limit_zh, 12, 600, T["ink"], "end"),
        text(1240, 702, limit_en, 11, 400, T["soft"], "end"),
    ])


def page(slug, title_short, desc, body, vb_w=1280, vb_h=720, lang="zh-Hant"):
    svg = (f'<svg viewBox="0 0 {vb_w} {vb_h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">'
           f'<title id="{slug}-title">{esc(title_short)}</title><desc id="{slug}-desc">{esc(desc)}</desc>'
           f'{defs()}<rect width="100%" height="100%" fill="{T["paper"]}"/>{body}</svg>')
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(title_short)}</title>
  <link href="{FONT_LINK}" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {T["paper"]}; color: {T["ink"]}; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; }}
    .frame {{ max-width: {vb_w + 40}px; width: 100%; }}
    /* min-width == the viewBox width so the type ramp is never scaled down; the wrapper scrolls, not the page. */
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    svg {{ width: 100%; min-width: {vb_w}px; display: block; }}
    @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="diagram-container">
      {svg}
    </div>
  </div>
</body>
</html>
"""


def write_outputs(slug, html_text, png=True, svg=True, scale=2):
    """Write <slug>.html, then export PNG (Playwright, the skill's export.md procedure) and SVG (skill helper)."""
    out_html = os.path.join(FIG_DIR, slug + ".html")
    with open(out_html, "w", encoding="utf-8", newline="\n") as f:
        f.write(html_text)
    print("wrote", out_html)
    if png:
        from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright
        import pathlib
        out_png = os.path.join(FIG_DIR, slug + ".png")
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge")
            except Exception:
                browser = p.chromium.launch()
            page_ = browser.new_page(device_scale_factor=scale)
            page_.goto(pathlib.Path(out_html).resolve().as_uri() + "?motion=static", wait_until="domcontentloaded")
            try:
                page_.wait_for_load_state("networkidle", timeout=20000)
            except PWTimeout:
                page_.evaluate("window.stop()")
                page_.wait_for_timeout(3000)
                print("warning: webfont request stalled - fallback typography", file=sys.stderr)
            page_.evaluate("document.fonts.ready")
            ok = page_.evaluate("() => ['12px Geist','12px \"Geist Mono\"','12px \"Noto Sans TC\"'].map(f => document.fonts.check(f))")
            print("fonts available (Geist, Geist Mono, Noto Sans TC):", ok)
            svg_el = page_.locator("svg").first
            svg_el.evaluate("""el => { const b = el.viewBox.baseVal;
                for (const [axis, size] of [['width', b.width], ['height', b.height]])
                    for (const prefix of ['', 'min-', 'max-']) el.style.setProperty(prefix + axis, size + 'px', 'important');
                for (let a = el.parentElement; a; a = a.parentElement) a.style.setProperty('overflow', 'visible', 'important'); }""")
            svg_el.screenshot(path=out_png)
            browser.close()
        print("wrote", out_png)
    if svg:
        out_svg = os.path.join(FIG_DIR, slug + ".svg")
        r = subprocess.run([sys.executable, os.path.join(SKILL, "scripts", "export_svg.py"), out_html, out_svg], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[:300])
    r = subprocess.run([sys.executable, os.path.join(SKILL, "scripts", "self_check.py"), out_html], capture_output=True, text=True)
    print("self_check:", (r.stdout + r.stderr).strip()[:600])
