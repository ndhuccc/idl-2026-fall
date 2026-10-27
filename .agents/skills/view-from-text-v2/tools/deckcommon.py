"""Shared helpers for the KP decks: one script builds the zh, en (or bilingual) deck, switched by KP_LANG."""
import json, os, re, sys
os.environ.setdefault("DECK_WORK", os.getcwd())
from pathlib import Path

def _skills_root():
    if os.environ.get("COURSE_SKILLS") or os.environ.get("SKILLS_DIR"):
        root = Path(os.environ.get("COURSE_SKILLS") or os.environ["SKILLS_DIR"]).expanduser().resolve()
    elif os.environ.get("COURSE_REPO"):
        root = Path(os.environ["COURSE_REPO"]).expanduser().resolve() / ".agents/skills"
    else:
        root = next((parent / ".agents/skills"
                     for start in (Path(__file__).resolve().parent, Path.cwd())
                     for parent in (start, *start.parents)
                     if (parent / ".agents/skills").is_dir()), None)
    if root is None or not (root / "view-from-text/tools/deckkit").is_dir():
        raise RuntimeError("Cannot locate course skills; set COURSE_REPO or COURSE_SKILLS")
    return root

sys.path.insert(0, str(_skills_root() / "view-from-text/tools/deckkit"))
import kit
from kit import AUX, EDGE, MAIN, OK, REF, WARN
VIO, ROSE = "#a98bd0", "#d08bb0"
kit.RGB[VIO] = (169, 139, 208); kit.RGB[ROSE] = (208, 139, 176)
TEAL = AUX
KP_LANG = os.environ.get("KP_LANG", "zh")          # zh | en | bi
RENAME = {}
TERM_EN = {}

def X(zh, en_):
    if KP_LANG == "zh":
        return zh
    if KP_LANG == "en":
        return en_
    return zh + "\n" + "\n".join("*" + ln + "*" if ln.strip() else ln for ln in en_.split("\n"))

def card(p, key, x, y, w, h, zh, en_, **kw):
    nid = p.card(key, x, y, w, h, X(zh, en_), **kw)
    if key.startswith("term:"): RENAME[nid] = key
    return nid

def bridge(p, zh, en_):
    h = p.need_h("bridge", 70)
    p.card("bridge", 60, 134, 1160, h, X(zh, en_), c=None, size=22)
    return 134 + h + 10

def stack(p, x, y, w, items, gap=10):
    """items: (key, zh, en, color, min_h)"""
    ids = []
    for key, zh, en_, c, mh in items:
        h = p.need_h(key, mh)
        ids.append(card(p, key, x, y, w, h, zh, en_, c=c, size=22))
        y += h + gap
    return y, ids

def newpage(u, lz, le, tz, te):
    if KP_LANG == "en":
        return u.page(le, te, "")
    return u.page(lz, tz, te if KP_LANG == "bi" else "")

def step(p, nz, ne):
    p.step(ne if KP_LANG == "en" else nz)

def reveal(p, ids, start=500, every=1400):
    for nid in ids:
        kit.nodes[nid]["initiallyHidden"] = True
    kit.steps[-1]["reveal"] = {"timing": "after-camera", "exitNodeIds": [],
                               "enter": [{"nodeId": nid, "effect": "fade", "delayMs": start + k * every, "durationMs": 500} for k, nid in enumerate(ids)]}

def finish(U, title_zh, title_en, note_zh, note_en, out):
    doc = kit.assemble(U, title_en if KP_LANG == "en" else title_zh, [], {"x": 200, "y": 100, "zoom": 1.2, "rotation": 0})
    for nid, nm in RENAME.items():
        doc["nodes"][nid]["name"] = TERM_EN.get(nm, nm) if KP_LANG == "en" else nm
    doc["steps"][0]["speakerNotes"] = note_en if KP_LANG == "en" else note_zh
    if KP_LANG != "bi":
        for nid in [k for k, n in doc["nodes"].items() if k.endswith("-sub") and not (n.get("text") or "").strip()]:
            sc = doc["nodes"][nid]["sceneId"]; doc["scenes"][sc]["nodeIds"].remove(nid); del doc["nodes"][nid]
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("pages", sum(1 for s in doc["steps"] if s["target"]["sceneId"] != "scene-root"), "KB", os.path.getsize(out) // 1024)

def md_rows(md):
    rows = []
    for ln in md.strip().split("\n"):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells): continue
        rows.append(cells)
    return rows

def table(p, key, x, y, w, md_zh, md_en, rowh=84, size=18, widths=None, gap=6):
    """Native table: one card per cell (the preview/HTML renderers do not draw Markdown tables). Returns the bottom y."""
    rows = md_rows(X(md_zh, md_en) if KP_LANG != "bi" else md_zh)
    n = len(rows[0]); widths = widths or [1.0 / n] * n
    cw = [int((w - gap * (n - 1)) * f) for f in widths]
    for r, row in enumerate(rows):
        cx = x
        for c, txt in enumerate(row):
            if txt:
                p.card(f"{key}-{r}-{c}", cx, y, cw[c], rowh, txt, c=(EDGE if r == 0 else None), size=size, align="center")
            cx += cw[c] + gap
        y += rowh + gap
    return y

def wide_page(u, t, rev=False, tb=None, tb_pos="right"):
    """Page without a figure (CLARITY C20/C21). Table: native cells, right of the cards (default) or below them."""
    p = newpage(u, *t["lab"], *t["title"]); y = bridge(p, *t["bridge"])
    if tb and tb_pos == "right":
        _, ids = stack(p, 60, y, 560, t["cards"])
        table(p, "tbl", 640, y, 580, tb[0], tb[1], rowh=92, size=22)
    elif tb:
        y2, ids = stack(p, 60, y, 1160, t["cards"])
        table(p, "tbl", 60, y2 + 4, 1160, tb[0], tb[1], rowh=64, size=22, widths=tb[2] if len(tb) > 2 else None)
    else:
        _, ids = stack(p, 60, y, 1160, t["cards"])
    step(p, *t["notes"])
    if rev: reveal(p, ids)
    return p

def fig_table_page(u, t, fig, tb, rev=False, widths=None, rowh=56):
    """Figure on the left; on the right the cards, then a native table."""
    p = newpage(u, *t["lab"], *t["title"]); y = bridge(p, *t["bridge"])
    p.figure(fig, 60, y)
    y2, ids = stack(p, 640, y, 580, t["cards"])
    table(p, "tbl", 640, y2 + 2, 580, tb[0], tb[1], rowh=rowh, size=22, widths=widths, gap=4)
    step(p, *t["notes"])
    if rev: reveal(p, ids)
    return p

