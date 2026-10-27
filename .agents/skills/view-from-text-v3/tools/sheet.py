"""Contact sheet of the HTML-designed pages (2 columns).
    python3 sheet.py <out_dir> <kpNN> <zh|en> p1 p2 p3 p4 sheet.png        e.g.  python3 sheet.py out kp13 zh p1 p2 p3 p4 s1.png
Reads <out_dir>/<kpNN>_<page>_<lang>.png written by `python3 kpNN_pages.py <out_dir>`."""
import sys
from PIL import Image
out_dir, kp, lang = sys.argv[1:4]; names = sys.argv[4:-1]; out = sys.argv[-1]
ims = [Image.open(f"{out_dir}/{kp}_{n}_{lang}.png") for n in names]; w, h = ims[0].size
S = Image.new("RGB", (w * 2, h * ((len(ims) + 1) // 2)), "#000")
for i, im in enumerate(ims): S.paste(im, ((i % 2) * w, (i // 2) * h))
S.save(out)
