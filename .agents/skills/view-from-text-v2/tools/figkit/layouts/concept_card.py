from ..parts import T, NFig, header_chip, panel, chip, takeaway, formula_bar, tint, solid, INK
from . import layout


@layout("concept_card", "learn one term or rule: the term, its plain definition, then labelled parts (example, non-example, formula, common mistake)")
def build(spec, lang=None):
    """spec: id, alt, term, definition (text or math), parts[{label, text, role}] (2..4), formula?, name? (term node name), takeaway.  Height 420."""
    f = NFig(spec["id"], 1160, 420, T(spec["alt"], lang))
    header_chip(f, 0, 0, 1160, 44, "blue", T(spec["term"], lang), size=24)
    panel(f, 0, 52, 1160, 96, "blue")
    f.text(24, 100, T(spec["definition"], lang), "TXT", 22, False, "left", w=1110, math=True, name=("term:" + T(spec["term"], lang)) if spec.get("name", True) else None)
    parts = spec["parts"]; n = len(parts); gap = 18; w = (1160 - gap * (n - 1)) / n
    for i, p in enumerate(parts):
        x = i * (w + gap)
        panel(f, x, 160, w, 150, p["role"])
        chip(f, x + 12, 170, 100, 28, p["role"], T(p["label"], lang), size=16, bold=True, alpha=0.4)
        f.text(x + 14, 252, T(p["text"], lang), "TXT", 18, False, "left", w=w - 28, math="$" in T(p["text"], lang))
    if spec.get("formula"):
        f.rect(0, 322, 1160, 44, fill=tint("blue", 0.14), radius=12)
        f.text(580, 344, spec["formula"], "TXT", 24, w=900, math=True)
    takeaway(f, 0, 374, 1160, 40, T(spec["takeaway"], lang))
    return f
