"""Scaffold a unit working folder:  python3 new_unit.py <dir> <KPNN> "<zh unit name>" "<en unit name>"
Set LEC=lec06 (default) for the IDL week prefix.
Creates <dir>/{KPNN_deck.py, KPNN_text.py (one example page), terms.json, terms_en.json, deckcommon.py, W_zh/, W_en/} ready for run_unit.sh."""
import os, sys, shutil
d, kp, zh, en = sys.argv[1:5]; kp = kp.lower(); LEC = os.environ.get("LEC", "lec06"); WEEK = int(LEC[3:]); HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.join(HERE, "..", "..", "view-from-text-v2", "tools")
os.makedirs(d + "/W_zh", exist_ok=True); os.makedirs(d + "/W_en", exist_ok=True); shutil.copy(os.path.join(V2, "deckcommon.py"), d)
open(f"{d}/W_zh/figs.json", "w").write("{}"); open(f"{d}/W_en/figs.json", "w").write("{}"); open(f"{d}/terms.json", "w").write("{}"); open(f"{d}/terms_en.json", "w").write("{}")
open(f"{d}/{kp}_deck.py", "w").write(f'''import os, sys
os.environ["DECK_PFX"] = "{LEC}{kp}"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
V2 = "{os.path.abspath(os.path.join(HERE, '..', '..', 'view-from-text-v2'))}"; sys.path.insert(0, os.path.join(V2, "tools"))
from deckv2 import run
from {kp}_text import PAGES, UNIT_ZH, UNIT_EN, TITLE_ZH, TITLE_EN, NOTE_ZH, NOTE_EN
if __name__ == "__main__":
    run(PAGES, UNIT_ZH, UNIT_EN, TITLE_ZH, TITLE_EN, NOTE_ZH, NOTE_EN, os.path.join(HERE, "draft.jbd.json"))
''')
open(f"{d}/{kp}_text.py", "w").write(f'''import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SK = "{os.path.abspath(os.path.join(HERE, '..', '..'))}"
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples"): sys.path.insert(0, os.path.join(SK, p))
UNIT_ZH, UNIT_EN = {zh!r}, {en!r}
TITLE_ZH = "深度學習導論 第{WEEK}週 {kp.upper()}：" + UNIT_ZH + "（大學部）"; TITLE_EN = "Introduction to Deep Learning, Week {WEEK}, {kp.upper()}: " + UNIT_EN + " (undergraduate)"
NOTE_ZH = "教材程度：大學部。"; NOTE_EN = "Level: undergraduate."
P = []
# one page = one dict; lab/title/bridge/notes are (zh, en) tuples, e.g. dict(lab=("標籤","Label"), title=(...), bridge=(...), notes=(...), layout=..., spec=...).
# one page = one dict.  freeform HTML page:  layout="freeform", spec=lambda: my_fig()   (my_fig returns an NFig, e.g. htmlkit.native(...))
# layout-library page: layout="plot_xy", spec=lambda: spec_dict   (see view-from-text-v2/LAYOUTS.md)
PAGES = P
''')
print("created", d)
