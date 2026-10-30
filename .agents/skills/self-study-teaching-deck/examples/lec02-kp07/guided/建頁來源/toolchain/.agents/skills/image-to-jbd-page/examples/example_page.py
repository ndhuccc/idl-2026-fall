"""Example: a small two-panel page built with jbd_page.Page, then put into a new deck.

    python3 example_page.py /tmp/demo        # writes /tmp/demo/demo.jbd.json (+ nodes.json); render with ../tools/compare.py or preview_deck.py

The source "picture" is imagined as 1600x900; every number below is a measured pixel of that picture."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TOOLS = os.path.join(HERE, "..", "tools"); sys.path.insert(0, TOOLS)
from jbd_page import Page
BLUE, RED, GREEN, GOLD = "#6aa8ff", "#ef6b6b", "#5fd08a", "#e8c85a"


def build():
    P = Page(src_w=1600, src_h=900)
    P.label(50, 10, 1500, 56, "Title in one centred line", size=40, color="#f2ede4")
    # left panel: a card with a 4x4 grid, an arrow and a curve
    P.rect(40, 100, 740, 640, fill="rgba(106,168,255,0.07)", stroke=BLUE, sw=3, radius=20)
    P.text(70, 112, 600, 40, "BEFORE", size=28, color=BLUE, bold=True)
    P.grid(80, 180, 4, 4, 48, fills={(0, 0): RED, (3, 3): BLUE})
    P.line([(300, 250), (420, 250), (420, 330)], GOLD, sw=4, arrow="end")          # corner polyline with an arrow head
    P.line([(80, 520), (200, 440), (320, 600), (440, 470), (560, 560)], GREEN, sw=5, smooth=True)   # smooth curve
    P.line([(80, 650), (700, 650)], "rgba(245,241,233,0.5)", sw=3, dash=True)        # dashed line
    # right panel: formula, ellipse badge, polygon
    P.rect(820, 100, 740, 640, fill="rgba(239,107,107,0.07)", stroke=RED, sw=3, radius=20)
    P.text(850, 112, 600, 40, "AFTER", size=28, color=RED, bold=True)
    P.text(850, 190, 680, 70, r"$g(\boldsymbol{x})=\sum_i w_i f_i(\boldsymbol{x})$", size=34, align="center")
    P.ellipse(900, 360, 28, fill=RED); P.text(950, 335, 560, 50, "badge + one text line", size=24)
    P.poly([(900, 460), (960, 520), (900, 580), (840, 520)], fill=GOLD)
    P.rect(1010, 470, 500, 110, fill="rgba(245,241,233,0.08)", stroke="rgba(245,241,233,0.4)", sw=2, radius=14)
    P.label(1010, 470, 500, 110, "Label centred in a box", size=26)
    # bottom banner
    P.rect(40, 770, 1520, 110, fill="rgba(30,58,160,0.55)", stroke="rgba(120,150,255,0.6)", sw=2, radius=18)
    P.label(60, 785, 1480, 80, "Conclusion banner", size=34, color="#f2ede4")
    return P


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_out"; os.makedirs(out, exist_ok=True)
    P = build(); P.dump(os.path.join(out, "nodes.json"))
    deck = os.path.join(out, "demo.jbd.json")
    subprocess.check_call([sys.executable, os.path.join(TOOLS, "new_deck.py"), deck, "--title", "Demo"])
    subprocess.check_call([sys.executable, os.path.join(TOOLS, "insert_page.py"), deck, "--nodes", os.path.join(out, "nodes.json"), "--after", "0", "--label", "Demo page", "--notes", "Speaker notes go here."])
