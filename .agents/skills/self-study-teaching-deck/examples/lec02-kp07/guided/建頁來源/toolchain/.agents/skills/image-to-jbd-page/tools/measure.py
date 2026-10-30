#!/usr/bin/env python3
"""measure.py -- read exact coordinates and colours from the source picture (needs Pillow).

    python3 measure.py grid src.png --step 50 --out grid.png        # picture with a labelled coordinate grid: read x/y of every element
    python3 measure.py crop src.png 40 120 760 600 --zoom 2 --out c.png   # enlarge a region (x y w h) to read small text and lines
    python3 measure.py colors src.png 80 160 700 300 ...            # hex colour at each (x y) point
    python3 measure.py palette src.png --n 10                       # dominant colours (to pick the fill / stroke tokens)
All coordinates are in the picture's own pixels -- the same numbers you pass to jbd_page.Page (it scales to 1280x720 for you)."""
import argparse
from collections import Counter
from PIL import Image, ImageDraw


def hexc(p): return "#%02x%02x%02x" % tuple(p[:3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid"); g.add_argument("src"); g.add_argument("--step", type=int, default=50); g.add_argument("--out", default="grid.png")
    c = sub.add_parser("crop"); c.add_argument("src"); [c.add_argument(k, type=int) for k in ("x", "y", "w", "h")]; c.add_argument("--zoom", type=float, default=2); c.add_argument("--out", default="crop.png")
    k = sub.add_parser("colors"); k.add_argument("src"); k.add_argument("xy", nargs="+", type=int)
    p = sub.add_parser("palette"); p.add_argument("src"); p.add_argument("--n", type=int, default=10)
    a = ap.parse_args(); im = Image.open(a.src).convert("RGB")
    if a.cmd == "grid":
        d = ImageDraw.Draw(im, "RGBA")
        for x in range(0, im.width, a.step):
            d.line([(x, 0), (x, im.height)], fill=(255, 0, 255, 90 if x % (a.step * 2) else 170), width=1); d.text((x + 2, 2), str(x), fill=(255, 0, 255, 255))
        for y in range(0, im.height, a.step):
            d.line([(0, y), (im.width, y)], fill=(0, 160, 255, 90 if y % (a.step * 2) else 170), width=1); d.text((2, y + 2), str(y), fill=(0, 160, 255, 255))
        im.save(a.out); print(f"{im.width}x{im.height} -> {a.out}")
    elif a.cmd == "crop":
        r = im.crop((a.x, a.y, a.x + a.w, a.y + a.h)); r = r.resize((int(r.width * a.zoom), int(r.height * a.zoom)), Image.LANCZOS); r.save(a.out); print(f"{r.width}x{r.height} -> {a.out}")
    elif a.cmd == "colors":
        for i in range(0, len(a.xy) - 1, 2): print(a.xy[i], a.xy[i + 1], hexc(im.getpixel((a.xy[i], a.xy[i + 1]))))
    else:
        q = im.quantize(colors=a.n, method=Image.MEDIANCUT).convert("RGB"); cnt = Counter(q.getdata()); tot = sum(cnt.values())
        for col, n in cnt.most_common(a.n): print(hexc(col), f"{100 * n / tot:.1f}%")


if __name__ == "__main__":
    main()
