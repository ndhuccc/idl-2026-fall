"""Demo: open polylines and curves in an HTML-designed page (converted to JianBauDau shapeKind "polyline"), next to a closed polygon.
    python3 polyline_demo.py <out_dir>      -> <out_dir>/polyline_demo_p1_{zh,en}.{html,png,json}"""
import sys, os
SK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # .agents/skills
for p in ("view-from-text-v3/tools", "view-from-text-v2/tools", "view-from-text-v2/examples", "view-from-text/tools/deckkit", "view-from-text-v3/examples"): sys.path.insert(0, os.path.join(SK, p))
from kp17_pages import card, svg, txt, TXT, build
def page(lang):
    s = ('<polyline points="60,380 200,200 340,330 480,150" fill="none" stroke="#e0c85f" stroke-width="6"/>'                                         # corners
         '<polyline points="560,380 700,200 840,330 980,150" fill="none" stroke="#7fb0e6" stroke-width="5" stroke-dasharray="12 8"/>'              # dashed
         '<path d="M60 440 C 200 300, 340 460, 480 340" fill="none" stroke="#e8896a" stroke-width="6"/>'                                          # open Bezier (no Z)
         '<polyline data-smooth="all" points="560,100 640,40 720,120 800,50 900,110" fill="none" stroke="#f9a8d4" stroke-width="7"/>'             # all curve points
         '<polygon points="640,420 700,360 760,420 700,470" fill="#4f9a68"/>')                                                                    # a closed polygon still fills
    return card(0, 0, 1160, 480, "blue", .1) + svg(0, 0, 1160, 480, s) + txt(20, 10, 600, "polyline demo", 18, TXT, True, "left", 28)
if __name__ == "__main__": build("polyline_demo_p1", page, 1160, 480, sys.argv[1])
