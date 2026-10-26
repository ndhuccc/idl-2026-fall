from ..parts import T, NFig, chip, takeaway, tint, solid, INK
from . import layout


def _leaves(n):
    return 1 if not n.get("children") else sum(_leaves(c["node"]) for c in n["children"])


@layout("tree_diagram", "a tree of questions, cases or derivation steps: nodes as chips, edges with labels (yes/no, conditions), leaves coloured by outcome")
def build(spec, lang=None):
    """spec: id, alt, root{text, role, children[{edge, node{...}}]}, takeaway.  Depth <= 4, <= 8 leaves.  Height 80 + 90*depth."""
    def depth(n): return 1 + max([depth(c["node"]) for c in n.get("children", [])] or [0])
    D = depth(spec["root"]); H = 60 + 96 * D + 44
    f = NFig(spec["id"], 1160, H, T(spec["alt"], lang))
    W = 1160; L = _leaves(spec["root"]); slot = W / L; pos = {}

    def place(n, lvl, left):
        k = _leaves(n); cx = left + k * slot / 2; y = 10 + lvl * 96
        if n.get("children"):
            x = left
            for c in n["children"]:
                w = _leaves(c["node"]) * slot; pos_child = place(c["node"], lvl + 1, x); x += w
                f.line(cx, y + 36, pos_child[0], pos_child[1], "MUT", 2)
                f.text((cx + pos_child[0]) / 2 + (12 if pos_child[0] > cx else -12), (y + 36 + pos_child[1]) / 2 - 4, T(c.get("edge", ""), lang), "MUT", 16, w=110)
        wch = max(120, min(slot - 20, 12 * len(T(n["text"], lang)) + 40))
        f.rect(cx - wch / 2, y, wch, 36, fill=solid(n.get("role", "blue")), radius=10, text=T(n["text"], lang), tcolor=INK, size=18, bold=True)
        return (cx, y)
    place(spec["root"], 0, 0)
    takeaway(f, 0, H - 40, 1160, 38, T(spec["takeaway"], lang))
    return f
