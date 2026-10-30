#!/usr/bin/env python3
"""Quality gate for a generated JianBauDau deck (.jbd.json made from page-to-teaching-diagram pages).

Checks (see SKILL.md, quality gates 6 and 8, and "課堂簡報製作流程"):
  1. the opening step shows a KP short title (not the default "Page");
  2. layer check: no text node whose centre lies inside a higher-zIndex, opaque filled shape of the same page;
  3. figure-first: per page, count figure elements (star, polygon, polyline, triangle, diamond, ellipse wider than 14 px);
     a page with fewer than --min-figures is reported (cards-only pages must be reworked).
Usage:  python3 check_deck_quality.py deck.jbd.json [--min-figures 4] [--page-width 1280]
Exit code 1 if check 1 or 2 fails; check 3 only warns (exit 0) unless --strict.
"""
import argparse, collections, json, re, sys

def alpha(c):
    if not c or c == "transparent": return 0.0
    m = re.match(r"rgba\(([^)]+)\)", c.strip())
    if m:
        p = [x.strip() for x in m.group(1).split(",")]
        return float(p[3]) if len(p) > 3 else 1.0
    return 1.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("deck"); ap.add_argument("--min-figures", type=int, default=4)
    ap.add_argument("--page-width", type=int, default=1280); ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(); d = json.load(open(a.deck, encoding="utf-8")); bad = 0
    steps = sorted(d["steps"], key=lambda s: s["order"])
    first = (steps[0].get("label") or "").strip() if steps else ""
    if not first or first.lower().startswith("page"): print("FAIL  opening step label is empty or 'Page':", repr(first)); bad += 1
    else: print("ok    opening step label:", first)
    pages = collections.defaultdict(list)
    for n in d["nodes"].values():
        if n["sceneId"] == d["rootSceneId"] or n["type"] == "connector": continue
        pages[(n["sceneId"], int(n["transform"]["x"] // a.page_width))].append(n)
    z_bad = []
    for key, ns in pages.items():
        fills = [n for n in ns if n["type"] == "shape" and alpha(n["style"].get("fill")) >= 0.9 and n.get("shapeKind") != "polyline"]
        for t in ns:
            if not (t.get("text") or "").strip(): continue
            cx = t["transform"]["x"] + t["width"] / 2; cy = t["transform"]["y"] + t["height"] / 2
            for f in fills:
                if f is t or f["zIndex"] <= t["zIndex"]: continue
                fx, fy = f["transform"]["x"], f["transform"]["y"]
                if fx <= cx <= fx + f["width"] and fy <= cy <= fy + f["height"]: z_bad.append((key, t["text"][:30], t["zIndex"], f["zIndex"])); break
    if z_bad:
        bad += 1; print("FAIL  text under a higher opaque shape: %d" % len(z_bad)); [print("      ", z) for z in z_bad[:10]]
    else: print("ok    no text hidden under higher shapes")
    low = []
    for key in sorted(pages):
        ns = pages[key]
        if len(ns) < 8: continue          # stray decoration outside the page columns
        fig = sum(1 for n in ns if n["type"] == "shape" and (n.get("shapeKind") in ("star", "polygon", "polyline", "triangle", "diamond") or (n.get("shapeKind") == "ellipse" and n["width"] > 14)))
        if fig < a.min_figures: low.append((key[1] + 1, fig))
    print(("WARN  pages with fewer than %d figure elements (page, count): %s" % (a.min_figures, low)) if low else "ok    every page has enough figure elements")
    if low and a.strict: bad += 1
    sys.exit(1 if bad else 0)
main()
