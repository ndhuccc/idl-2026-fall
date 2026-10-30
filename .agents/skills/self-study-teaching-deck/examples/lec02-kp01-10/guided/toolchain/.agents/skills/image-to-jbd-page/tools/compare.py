#!/usr/bin/env python3
"""compare.py -- render one page of a deck and put it next to the source picture, so you can see what differs.

    python3 compare.py deck.jbd.json 3 src.png --out cmp.png          # page 3 of the deck vs the picture
    python3 compare.py deck.jbd.json 3 src.png --out cmp.png --overlay   # also a 50% blend (misaligned elements show as doubles)

Needs Playwright (+ Chromium) and Pillow; KaTeX comes from a CDN (set KATEX_DIR for offline use, see preview_deck.py).
Prints the text-overflow report of the render. Read cmp.png with an image viewer and fix the largest visual differences first:
layout and positions, then colours, then text sizes. The preview is an approximation of JianBauDau, not the app itself."""
import argparse, os, subprocess, sys, tempfile
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck"); ap.add_argument("page", type=int); ap.add_argument("src"); ap.add_argument("--out", default="compare.png"); ap.add_argument("--overlay", action="store_true")
    a = ap.parse_args(); tmp = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(HERE, "preview_deck.py"), a.deck, "--out", tmp], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-600:])
    pngs = sorted(f for f in os.listdir(tmp) if f.endswith(f"_p{a.page}.png"))
    if not pngs: raise SystemExit(f"page {a.page} was not rendered")
    out = Image.open(os.path.join(tmp, pngs[0])).convert("RGB"); src = ImageOps.pad(Image.open(a.src).convert("RGB"), out.size, method=Image.LANCZOS, color="#151513", centering=(0.5, 0.5))
    n = 3 if a.overlay else 2; sheet = Image.new("RGB", (out.width * n + 10 * (n - 1), out.height), (60, 60, 60))
    sheet.paste(src, (0, 0)); sheet.paste(out, (out.width + 10, 0))
    if a.overlay: sheet.paste(Image.blend(src, out, 0.5), (2 * out.width + 20, 0))
    sheet.save(a.out); out.save(a.out.replace(".png", "_render.png")); print("left = source, right = render ->", a.out)


if __name__ == "__main__":
    main()
