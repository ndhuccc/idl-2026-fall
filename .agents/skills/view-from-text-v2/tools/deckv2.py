"""deckv2.py -- build a deck from a page list. Figure pages come from layouts; text pages from cards (the v1 helpers).

    # kpNN_deck.py
    import os; os.environ["DECK_PFX"] = "lec05kp12"          # before importing deckv2
    from deckv2 import run
    from kpNN_text import PAGES, UNIT_ZH, UNIT_EN, TITLE_ZH, TITLE_EN, NOTE_ZH, NOTE_EN
    if __name__ == "__main__": run(PAGES, UNIT_ZH, UNIT_EN, TITLE_ZH, TITLE_EN, NOTE_ZH, NOTE_EN)

A page is a dict: lab=(zh,en), title=(zh,en), bridge=(zh,en), notes=(zh,en), and either
  layout="cards", cards=[(key, zh, en, colour, min_h), ...]  [, tb=(md_zh, md_en)]           -- text page
  layout=<registered layout name>, spec=<dict or zero-argument callable>                      -- layout figure page
  layout="freeform", spec=<zero-argument callable returning an NFig>                          -- hand-drawn figure page
Run with KP_LANG=zh|en (the build helpers in ../run scripts do both)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deckcommon import *  # noqa: E402,F401,F403
import deckcommon as _D  # noqa: E402
from figkit import layouts  # noqa: E402


def build(PAGES, unit_zh, unit_en):
    u = kit.Unit("u01", unit_en if _D.KP_LANG == "en" else unit_zh, 0, 0)
    for t in PAGES:
        lay = t["layout"]
        if lay == "cards":
            wide_page(u, t, rev=t.get("rev", False), tb=t.get("tb"), tb_pos=t.get("tb_pos", "right"))
            continue
        p = newpage(u, *t["lab"], *t["title"]); y = bridge(p, *t["bridge"])
        spec = t["spec"]() if callable(t["spec"]) else t["spec"]
        fig = spec if lay == "freeform" else layouts.build(lay, spec)
        fig.to_nodes(p, 60, y, "dark")
        step(p, *t["notes"])
    return [u]


def run(PAGES, unit_zh, unit_en, title_zh, title_en, note_zh, note_en, out=None):
    U = build(PAGES, unit_zh, unit_en)
    finish(U, title_zh, title_en, note_zh, note_en, out or os.path.join(os.getcwd(), "draft.jbd.json"))
