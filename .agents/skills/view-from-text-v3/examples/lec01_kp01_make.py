import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lec01_kp01_pages as pages, lec01_kit as kit
from lec01_kit import build_deck
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for lang in ("zh", "en"):
    ps = pages.make(lang)
    for g, _, _ in ps: g.P.check()
    t = "KP01 選模型要有評分標準" if lang == "zh" else "KP01 Model selection needs a scoring rule"
    build_deck(ps, t, t, lang, f"{out}/lec01-kp01-model-selection-framework{'' if lang=='zh' else '-en'}.jbd.json")
