"""KP01 圖解共用:Page 代理(記錄 beat)、吉祥物、星形、組檔。"""
import sys, json, math, copy, re
import os
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "image-to-jbd-page", "tools")  # .agents/skills/image-to-jbd-page/tools
sys.path.insert(0, IMG)
from jbd_page import Page
import new_deck

BG, CARD, INK, MUTED = "#f8fafc", "#ffffff", "#1f2937", "#475569"
BLUE, BLUEL = "#1d4ed8", "#dbeafe"; ORG, ORGL = "#c2410c", "#ffedd5"; GRN, GRNL = "#15803d", "#dcfce7"
RED, REDL = "#b91c1c", "#fee2e2"; PUR, PURL = "#6d28d9", "#ede9fe"; BAN = "#1e3a8a"; GOLD = "#f59e0b"; GRAY = "#94a3b8"

class PG:
    """Wrap Page: every node created while beat=k is tagged k (0 = always visible)."""
    def __init__(self):
        self.P = Page(src_w=1280, src_h=720, prefix="k1"); self.beat = 0; self.tags = []; self.notes = {}; self.links = []
    def __getattr__(self, name):
        f = getattr(self.P, name)
        if not callable(f): return f
        def w(*a, **k):
            n0 = len(self.P.nodes); r = f(*a, **k)
            self.tags += [self.beat] * (len(self.P.nodes) - n0); return r
        return w

    def mark(self):
        """Index the next created node will have (use it to name a shape for link())."""
        return len(self.P.nodes)
    def link(self, a, b, ha, hb, color=GRAY, sw=4):
        """Native connector from node index a (hook ha) to node index b (hook hb); hooks 0-15 run clockwise from the top (4 right, 8 bottom, 12 left)."""
        self.links.append((a, b, ha, hb, color, sw, self.beat))

def star(P, cx, cy, R, fill, z=6, stroke="transparent", sw=0):
    pts = []
    for i in range(10):
        r = R if i % 2 == 0 else R * 0.45; a = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return P.poly(pts, fill=fill, stroke=stroke, sw=sw, z=z)

def tick(P, x, y, s, color=GRN, z=6):
    return P.line([(x, y + 0.55 * s), (x + 0.35 * s, y + s), (x + s, y)], color, sw=max(3, s * 0.18), z=z)

def cross(P, x, y, s, color=RED, z=6):
    P.line([(x, y), (x + s, y + s)], color, sw=max(3, s * 0.18), z=z); P.line([(x + s, y), (x, y + s)], color, sw=max(3, s * 0.18), z=z)

def mascot(P, x, y, s, mood="proud", z=5):
    """Robot 'little model'. (x, y) = top-left of the head; s = scale. Width about 130*s, height about 230*s."""
    X = lambda v: x + v * s; Y = lambda v: y + v * s
    P.line([(X(60), Y(0)), (X(60), Y(-26))], BLUE, sw=4 * s + 1, z=z)
    P.ellipse(X(60), Y(-30), 9 * s, fill=GOLD if mood == "proud" else ORG, z=z)
    P.rect(X(0), Y(0), 120 * s, 100 * s, fill=BLUEL, stroke=BLUE, sw=4 * s + 1, radius=26 * s, z=z)
    P.rect(X(14), Y(22), 92 * s, 52 * s, fill=CARD, stroke=BLUE, sw=2, radius=16 * s, z=z + 1)
    if mood == "proud":
        P.ellipse(X(38), Y(46), 9 * s, fill=INK, z=z + 2); P.ellipse(X(82), Y(46), 9 * s, fill=INK, z=z + 2)
        P.ellipse(X(41), Y(42), 3 * s, fill=CARD, z=z + 3); P.ellipse(X(85), Y(42), 3 * s, fill=CARD, z=z + 3)
        P.line([(X(40), Y(64)), (X(60), Y(74)), (X(80), Y(64))], INK, sw=3 * s + 1, smooth=True, z=z + 2)
        P.ellipse(X(24), Y(62), 6 * s, fill="#fda4af", z=z + 2); P.ellipse(X(96), Y(62), 6 * s, fill="#fda4af", z=z + 2)
    else:
        P.ellipse(X(38), Y(46), 8 * s, fill=INK, z=z + 2); P.ellipse(X(82), Y(46), 11 * s, fill=INK, z=z + 2)
        P.line([(X(26), Y(32)), (X(48), Y(28))], INK, sw=3 * s + 1, z=z + 2)
        P.line([(X(40), Y(66)), (X(50), Y(60)), (X(60), Y(66)), (X(70), Y(60)), (X(80), Y(66))], INK, sw=3 * s + 1, smooth=True, z=z + 2)
        P.poly([(X(112), Y(18)), (X(122), Y(40)), (X(112), Y(46)), (X(102), Y(40))], fill="#38bdf8", z=z + 2)
    P.rect(X(14), Y(106), 92 * s, 78 * s, fill=CARD, stroke=BLUE, sw=4 * s + 1, radius=20 * s, z=z)
    P.ellipse(X(60), Y(145), 14 * s, fill=PURL, stroke=PUR, sw=2, z=z + 1)
    if mood == "proud":
        P.line([(X(14), Y(124)), (X(-20), Y(96))], BLUE, sw=7 * s + 1, z=z); P.line([(X(106), Y(124)), (X(140), Y(96))], BLUE, sw=7 * s + 1, z=z)
        star(P, X(150), Y(80), 14 * s + 4, GOLD, z=z)
    else:
        P.line([(X(14), Y(130)), (X(-14), Y(158))], BLUE, sw=7 * s + 1, z=z); P.line([(X(106), Y(130)), (X(134), Y(158))], BLUE, sw=7 * s + 1, z=z)
    P.rect(X(26), Y(188), 28 * s, 30 * s, fill=BLUE, radius=8 * s, z=z); P.rect(X(66), Y(188), 28 * s, 30 * s, fill=BLUE, radius=8 * s, z=z)

def build_deck(pages, title, label0, lang, out):
    """pages = [(PG, title_label, notes{beat: text})]; write deck: page i at x = i * 1440 inside scene-u01."""
    d = new_deck.make(title, name=title)
    d["nodes"]["frame-u01"]["name"] = title; d["nodes"]["frame-u01"]["text"] = title
    d["scenes"]["scene-root"]["name"] = title; d["scenes"]["scene-u01"]["name"] = title
    d["steps"][0]["label"] = label0
    sc = d["scenes"]["scene-u01"]; order = 1; N = len(pages)
    for i, (pg, lab, notes) in enumerate(pages):
        ox = i * 1440; ids = {}
        for j, (n, b) in enumerate(zip(copy.deepcopy(pg.P.nodes), pg.tags)):
            if n["type"] == "text" and ("×" in n["text"] or "→" in n["text"]):      # math symbols belong in KaTeX
                segs = re.split(r"(\$[^$]*\$)", n["text"])
                n["text"] = "".join(sg if sg.startswith("$") else sg.replace("×", "$\\times$").replace("→", "$\\to$") for sg in segs)
            if n["type"] == "text" and "$" in n["text"] and n["height"] <= 48:      # KaTeX line boxes are a few px taller than the font size
                n["height"] = round(n["height"] + 8, 2); n["transform"]["y"] = round(n["transform"]["y"] - 4, 2)
                fs = n["style"]["fontSize"]; n["style"]["lineHeight"] = round(n["height"] / fs, 3) if n["style"]["lineHeight"] and n["style"]["lineHeight"] * fs >= n["height"] - 12 else n["style"]["lineHeight"]
            n["id"] = f"k1p{i+1}-{j+1}"; n["sceneId"] = "scene-u01"; n["transform"]["x"] = round(n["transform"]["x"] + ox, 2)
            if b > 0: n["initiallyHidden"] = True
            d["nodes"][n["id"]] = n; sc["nodeIds"].append(n["id"]); ids.setdefault(b, []).append(n["id"])
        for ci, (a_, b2, ha, hb, col, sw, bt) in enumerate(pg.links):
            cid = f"k1p{i+1}-c{ci + 1}"
            d["nodes"][cid] = {"id": cid, "type": "connector", "name": "connector", "sceneId": "scene-u01", "transform": {"x": ox, "y": 0, "scaleX": 1, "scaleY": 1, "rotation": 0}, "width": 1, "height": 1, "zIndex": 20, "locked": False,
                               "style": {"fill": "transparent", "stroke": col, "textColor": INK, "fontSize": 16, "radius": 0, "borderWidth": sw}, "text": "",
                               "connector": {"sourceNodeId": f"k1p{i+1}-{a_+1}", "sourceHook": ha, "targetNodeId": f"k1p{i+1}-{b2+1}", "targetHook": hb, "curve": False, "direction": "forward", "arrowSize": "medium"}}
            if bt > 0: d["nodes"][cid]["initiallyHidden"] = True
            sc["nodeIds"].append(cid); ids.setdefault(bt, []).append(cid)
        texts_by_beat = {}
        for n_, b_ in zip(pg.P.nodes, pg.tags):
            if n_["type"] == "text" and n_["text"].strip(): texts_by_beat.setdefault(b_, []).append(n_["text"].strip())
        cam = {"kind": "scene-point", "sceneId": "scene-u01", "point": {"x": ox + 640, "y": 360.0}, "zoom": 1.0, "referenceViewport": {"width": 1280, "height": 720}}
        d["steps"].append({"id": f"scene-u01-step-p{i+1}", "order": order, "label": f"{lab} · {i+1}/{N}", "durationMs": 900, "holdMs": 500, "easing": "smooth", "target": cam, "speakerNotes": notes.get(0, "")}); order += 1
        for b in sorted(k for k in ids if k > 0):
            d["steps"].append({"id": f"scene-u01-p{i+1}b{b}", "order": order, "label": f"{lab} · {i+1}/{N} ({b})", "durationMs": 900, "holdMs": 500, "easing": "smooth", "target": cam,
                               "reveal": {"timing": "after-camera", "enter": [{"nodeId": nid, "effect": "fade", "delayMs": 0, "durationMs": 450} for nid in ids[b]], "exitNodeIds": []},
                               "speakerNotes": notes.get(b, "") or " ".join(t.replace("$", "").replace("\n", " ") for t in texts_by_beat.get(b, []))}); order += 1
    json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d


# ---------------------------------------------------------------- shared page parts (all KP decks)
def new_page(n, N, kp, title):
    g = PG()
    g.rect(0, 0, 1280, 720, fill=BG, z=0)
    wid = sum(1.0 if ord(c) > 255 else 0.55 for c in title)      # CJK ~1 em, Latin ~0.55 em
    g.text(50, 22, 1010, 72, title, size=max(24, min(36, int(1000 / max(wid, 1)))), color=INK, bold=True, vcenter=True, z=8)
    g.text(1060, 36, 170, 30, f"{kp} · {n}/{N}", size=18, color=MUTED, align="right", z=8)
    return g

def banner(g, s):
    g.rect(50, 640, 1180, 60, fill=BAN, radius=14, z=3, shadowColor="rgba(30,58,138,0.35)", shadowBlur=12, shadowOffsetY=4)
    g.text(70, 640, 1140, 60, s, size=24, color="#ffffff", bold=True, align="center", vcenter=True, z=8)

def pill(g, x, y, w, s, fill, color="#ffffff", size=18):
    g.rect(x, y, w, 30, fill=fill, radius=15, z=6); g.text(x, y, w, 30, s, size=size, color=color, bold=True, align="center", vcenter=True, z=8)

def card(g, x, y, w, h, color=GRAY, sw=3, fill=CARD):
    g.rect(x, y, w, h, fill=fill, stroke=color, sw=sw, radius=16, z=1, shadowColor="rgba(15,23,42,0.12)", shadowBlur=14, shadowOffsetY=4)

def head(g, x, y, w, s, color, size=22):
    g.text(x, y, w, 44, s, size=size, color=color, bold=True, vcenter=True, z=8)

def brackets(g, x, y, w, h, color=INK, sw=4, z=5):
    """Matrix brackets [ ] around the box."""
    g.line([(x + 12, y), (x, y), (x, y + h), (x + 12, y + h)], color, sw=sw, z=z); g.line([(x + w - 12, y), (x + w, y), (x + w, y + h), (x + w - 12, y + h)], color, sw=sw, z=z)

def arrow(g, a, b, color=GRAY, sw=5, head=18, z=6, dash=False):
    g.line([a, b], color, sw=sw, arrow="end", head=head, z=z, dash=dash)

def run(make, out, kp_zh, kp_en, stem):
    os.makedirs(out, exist_ok=True)
    for lang in ("zh", "en"):
        ps = make(lang)
        for g, _, _ in ps: g.P.check()
        t = kp_zh if lang == "zh" else kp_en
        build_deck(ps, t, t, lang, f"{out}/{stem}{'' if lang == 'zh' else '-en'}.jbd.json")

