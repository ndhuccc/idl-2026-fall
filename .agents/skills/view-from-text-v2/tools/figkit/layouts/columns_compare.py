from ..parts import T, NFig, header_chip, panel, chip, outline_box, icon, formula_bar, takeaway, legend, tint, solid
from . import layout


@layout("columns_compare", "N side-by-side cases, each: example drawn with icons, substitution, big result, one comment; then a formula bar and a takeaway")
def build(spec, lang=None):
    """spec: id, alt, legend[(kind,text)], columns[{role, header, icons[(kind,n)], icons_mode 'rows'|'line', lines[bold, plain],
    eq[latex...], result latex, comment, note latex?}], formula latex, formula_caption, takeaway. Height 452."""
    f = NFig(spec["id"], 1160, 452, T(spec["alt"], lang))
    cols = spec["columns"]; n = len(cols); gap = 22; cw = (1160 - gap * (n - 1)) / n
    legend(f, 740, 14, [(k, T(t, lang)) for k, t in spec["legend"]], col_w=140)
    for k, c in enumerate(cols):
        x = k * (cw + gap); role = c["role"]
        header_chip(f, x, 34, cw, 38, role, T(c["header"], lang))
        panel(f, x, 78, cw, 276, role)
        outline_box(f, x + 12, 88, cw - 24, 96)
        rows = c["icons"]
        if c.get("icons_mode", "rows") == "line":
            seq = [kd for kd, m in rows for _ in range(m)]
            for j, kd in enumerate(seq):
                icon(f, kd, x + 34 + j * (28 if len(seq) > 3 else 40), 136)
        else:
            top = 88 + 48 - (len(rows) - 1) * 13
            for r, (kd, m) in enumerate(rows):
                for j in range(m):
                    icon(f, kd, x + 30 + j * 28, top + r * 26)
        f.text(x + 168, 118, T(c["lines"][0], lang), "TXT", 18, True, "left")
        f.text(x + 168, 150, T(c["lines"][1], lang), "TXT", 16, False, "left")
        for j, eq in enumerate(c.get("eq", [])):
            f.text(x + cw / 2, 204 + 28 * j, eq, "TXT", 18, w=cw - 20, math=True)
        f.rect(x + 76, 248, cw - 152, 58, fill=tint(role, 0.30), radius=12, text=c["result"], tcolor="TXT", size=26, bold=True, math=True)
        f.text(x + cw / 2, 320, T(c["comment"], lang), "TXT", 16, w=cw - 16)
        if c.get("note"):
            f.text(x + cw / 2, 342, T(c["note"], lang), "MUT", 16, w=cw - 16, math=True)
    f.rect(0, 362, 1160, 44, fill=tint("blue", 0.14), radius=12)
    f.text(250, 382, spec["formula"], "TXT", 22, w=360, math=True)
    f.text(600, 384, T(spec["formula_caption"], lang), "TXT", 18, w=540, ha="left", math=True)
    takeaway(f, 0, 412, 1160, 40, T(spec["takeaway"], lang))
    return f
