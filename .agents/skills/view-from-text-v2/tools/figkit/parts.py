"""parts.py -- design tokens and reusable parts for native infographics (view-from-text-v2).

Everything draws on an `nativefig.NFig`. Parts never hard-code a topic: they take roles (colour names), icon kinds and strings.
A string may be plain text or {"zh": ..., "en": ...}; `T(x, lang)` picks one (lang defaults to env KP_LANG, "zh").
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "view-from-text", "tools", "deckkit")))   # nativefig lives in v1's deckkit
from nativefig import NFig  # noqa: E402,F401

INK = "#1c1712"
# role -> (solid colour for bright fills with dark text, rgb for tints)
ROLES = {
    "red": ("#e07a70", (224, 122, 112)), "blue": ("#6a9fd6", (106, 159, 214)), "green": ("#7dbb8c", (125, 187, 140)),
    "orange": ("#d97757", (224, 150, 95)), "violet": ("#a98bd0", (169, 139, 208)), "yellow": ("#e0c85f", (224, 200, 95)),
    "teal": ("#5fb8a8", (95, 184, 168)), "grey": ("#b7b0a5", (183, 176, 165)),
}
# icon kinds -> colour (class 1 circle blue, class 2 square orange, class 3 triangle violet)
ICON_COLOR = {"circle": "#6a9fd6", "square": "#d97757", "tri": "#a98bd0"}


def lang():
    return "en" if os.environ.get("KP_LANG", "zh") == "en" else "zh"


def T(x, lg=None):
    if isinstance(x, dict):
        return x[lg or lang()]
    return x


def tr_deep(x, lg=None):
    """Resolve every {"zh","en"} pair inside lists/dicts."""
    if isinstance(x, dict) and set(x) == {"zh", "en"}:
        return T(x, lg)
    if isinstance(x, (list, tuple)):
        return [tr_deep(i, lg) for i in x]
    if isinstance(x, dict):
        return {k: tr_deep(v, lg) for k, v in x.items()}
    return x


def solid(role):
    return ROLES[role][0]


def tint(role, a=0.12):
    r, g, b = ROLES[role][1]
    return f"rgba({r},{g},{b},{a})"


# ---- boxes -------------------------------------------------------------------------------------------------------------------------
def header_chip(f, x, y, w, h, role, text, size=20):
    f.rect(x, y, w, h, fill=solid(role), radius=12, text=text, tcolor=INK, size=size, bold=True)


def panel(f, x, y, w, h, role):
    f.rect(x, y, w, h, fill=tint(role, 0.12), stroke=tint(role, 0.65), sw=2, radius=14)


def chip(f, x, y, w, h, role, text, size=16, bold=False, alpha=0.28, math=False, dark=False):
    """A small rounded box with one line of text. dark=True: bright solid fill with dark text (for verdicts such as 'pick B')."""
    if dark:
        f.rect(x, y, w, h, fill=solid(role), radius=10, text=text, tcolor=INK, size=size, bold=True, math=math)
    else:
        f.rect(x, y, w, h, fill=tint(role, alpha), radius=10, text=text, tcolor="TXT", size=size, bold=bold, math=math)


def outline_box(f, x, y, w, h, color="MUT", sw=1.5, radius=10):
    f.rect(x, y, w, h, stroke=color, sw=sw, radius=radius)


def formula_bar(f, x, y, w, h, formula, caption, split=0.25):
    f.rect(x, y, w, h, fill=tint("blue", 0.14), radius=12)
    f.text(x + w * split, y + h / 2, formula, "TXT", 22, w=360, math=True)
    f.text(x + w * 0.52, y + h / 2 + 2, caption, "TXT", 18, w=w * 0.46, ha="left", math=True)


def takeaway(f, x, y, w, h, text, size=20):
    f.rect(x, y, w, h, fill=tint("yellow", 0.16), stroke=tint("yellow", 0.55), sw=1.5, radius=12, text=text, tcolor="TXT", size=size, bold=True)


# ---- icons ---------------------------------------------------------------------------------------------------------------------------
def icon(f, kind, cx, cy):
    c = ICON_COLOR[kind]
    if kind == "circle":
        f.ellipse(cx, cy, 17, c)
    elif kind == "square":
        f.square(cx, cy, 15, c)
    else:
        f.tri(cx, cy, 17, c)


def icon_rows(f, x, y, rows, step=30, rowgap=32):
    """rows: [(kind, n), ...] one row per entry; (x, y) is the centre of the first icon row's first icon minus 15."""
    for r, (kind, n) in enumerate(rows):
        for k in range(n):
            icon(f, kind, x + 15 + k * step, y + r * rowgap)


def legend(f, x, y, items, col_w=140, vertical=False, row_h=30):
    for i, (kind, text) in enumerate(items):
        cx, cy = (x, y + i * row_h) if vertical else (x + i * col_w, y)
        icon(f, kind, cx, cy)
        f.text(cx + 20, cy, text, "TXT", 16, False, "left")


def leg_items(lg=None):
    return None


# ---- grids (images, kernels, feature maps) and glossary strips ---------------------------------------------------------------------
def grid(f, x, y, rows, cols, cell, cells=None, role="orange", show=False, scale=None, window=None, numfmt="{:g}"):
    """Draw a rows x cols grid with its top-left corner at (x, y). Cost: one outline + (rows-1)+(cols-1) lines + one box per non-zero cell.
    cells: matrix of numbers (None = empty grid). Non-zero cells are filled with the role colour, stronger for larger values (relative to
    `scale`, default the maximum); show=True prints the number in every cell. window=(r, c, h, w): a thick outline over a sub-block."""
    cells = cells or [[0] * cols for _ in range(rows)]
    sc = scale or max(max(abs(v) for v in r) for r in cells) or 1
    for r in range(rows):
        for c in range(cols):
            v = cells[r][c]
            if v:
                f.rect(x + c * cell, y + r * cell, cell, cell, fill=tint(role, 0.25 + 0.65 * min(abs(v) / sc, 1)))
    f.rect(x, y, cols * cell, rows * cell, stroke="MUT", sw=1.5)
    for r in range(1, rows):
        f.line(x, y + r * cell, x + cols * cell, y + r * cell, "MUT", 1)
    for c in range(1, cols):
        f.line(x + c * cell, y, x + c * cell, y + rows * cell, "MUT", 1)
    if show:
        for r in range(rows):
            for c in range(cols):
                f.text(x + c * cell + cell / 2, y + r * cell + cell / 2, numfmt.format(cells[r][c]), "TXT", 16, bool(cells[r][c]), w=cell)
    if window:
        wr, wc, wh, ww = window
        f.rect(x + wc * cell - 1, y + wr * cell - 1, ww * cell + 2, wh * cell + 2, stroke=solid("yellow"), sw=3)
    return x + cols * cell, y + rows * cell


def ring(f, cx, cy, r, color="MUT", sw=2):
    f.rect(cx - r, cy - r, 2 * r, 2 * r, stroke=color, sw=sw, radius=r)


def terms_strip(f, y, terms, h=76, role="blue"):
    """terms: [(name, definition)]. Each entry is one text node named 'term:<name>' (the term audit looks for it)."""
    n = len(terms); gap = 14; w = (1160 - gap * (n - 1)) / n
    for i, (nm, df) in enumerate(terms):
        x = i * (w + gap)
        f.rect(x, y, w, h, fill=tint(role, 0.12), stroke=tint(role, 0.5), sw=1.5, radius=12)
        f.text(x + 12, y + 15, nm, "TXT", 18, True, "left", w=w - 20)
        f.text(x + 12, y + (54 if "$" in df else 44), df, "TXT", 16, False, "left", w=w - 20, math="$" in df, name="term:" + nm)


# ---- sketches (small drawings used inside panels) ---------------------------------------------------------------------------------
def tree_shallow(f, cx, y, root_text, leaf_text, role_root="blue", role_leaf=("blue", "orange")):
    f.rect(cx - 55, y, 110, 32, fill=solid(role_root), radius=9, text=root_text, tcolor=INK, size=18, bold=True)
    for dx, role in zip((-120, 120), role_leaf):
        f.line(cx, y + 32, cx + dx, y + 76, "MUT", 2.5)
        f.rect(cx + dx - 55, y + 76, 110, 32, fill=solid(role), radius=9, text=leaf_text, tcolor=INK, size=18, bold=True)


def tree_deep(f, cx, y, levels=(80, 40, 20)):
    xs = [cx]
    f.rect(cx - 22, y, 44, 14, fill=solid("blue"), radius=5)
    for lvl, dx in enumerate(levels):
        ny = y + 34; nxs = []
        for px in xs:
            for sgn in (-1, 1):
                x = px + sgn * dx; nxs.append(x); f.line(px, y + 14, x, ny, "MUT", 1.5)
                f.rect(x - 16, ny, 32, 14, fill=solid("blue") if (lvl < len(levels) - 1 or sgn < 0) else solid("orange"), radius=5)
        xs, y = nxs, ny
    return y + 14


def magnifier_note(f, cx, cy, text_lines, bubble=(1004, 140, 148, 50)):
    f.ellipse(cx, cy, 54, "rgba(106,159,214,0.35)"); f.ellipse(cx, cy, 14, "#e0705a"); f.line(cx + 20, cy + 22, cx + 42, cy + 46, "TXT", 6)
    bx, by, bw, bh = bubble
    f.rect(bx, by, bw, bh, fill=tint("red", 0.28), radius=12)
    for i, ln in enumerate(text_lines):
        f.text(bx + bw / 2, by + 16 + i * 21, ln, "TXT", 16, w=bw - 2)


def grid_tags(f, cx, y, n, tag, role="blue", cell=34, caption=None):
    """n x n small squares, each carrying the same tag (e.g. the number of parameters it owns)."""
    x0 = cx - n * cell / 2
    for r in range(n):
        for c in range(n):
            f.rect(x0 + c * cell + 2, y + r * cell + 2, cell - 4, cell - 4, fill=tint(role, 0.35), radius=6, text=tag, tcolor="TXT", size=16, bold=True)
    if caption:
        f.text(cx, y + n * cell + 18, caption, "TXT", 16, w=n * cell + 120)


SKETCHES = {"grid_tags": grid_tags, "tree_shallow": tree_shallow, "tree_deep": tree_deep, "magnifier_note": magnifier_note}
