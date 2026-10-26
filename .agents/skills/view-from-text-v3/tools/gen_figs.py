"""Write the lecture PNGs (light theme) of a unit from the same specs as the deck.
    python3 gen_figs.py <specs_module_file.py> <map.json> <outdir>     map.json: {"function_name": "figure-file-stem", ...}
e.g. {"pnn": "lec05-kp18-arch"}  ->  outdir/lec05-kp18-arch.png and lec05-kp18-arch-en.png.   A spec function returns an NFig or (layout_name, spec)."""
import os, sys, json, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); SK = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples", "view-from-text/tools/deckkit"): sys.path.insert(0, os.path.join(SK, p))
mod_path, map_path, out = sys.argv[1:4]
from figkit import layouts
from figkit.parts import NFig
for lang in ("zh", "en"):
    os.environ["KP_LANG"] = lang
    spec = importlib.util.spec_from_file_location("specs_" + lang, mod_path); m = importlib.util.module_from_spec(spec); sys.path.insert(0, os.path.dirname(os.path.abspath(mod_path))); spec.loader.exec_module(m)
    for fn, stem in json.load(open(map_path)).items():
        r = getattr(m, fn)(); f = r if isinstance(r, NFig) else layouts.build(*r); f.to_png(os.path.join(out, stem + ("-en" if lang == "en" else "") + ".png"))
print("done")
