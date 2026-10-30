#!/usr/bin/env python3
"""Quality gate for a generated JianBauDau deck (.jbd.json made from page-to-teaching-diagram pages).

Checks (see SKILL.md, quality gates 6 and 8, and "課堂簡報製作流程"):
  1. the opening step shows a KP short title (not the default "Page");
  2. layer check: no text node whose centre lies inside a higher-zIndex, opaque filled shape of the same page;
  3. figure-first: per page, count figure elements (star, polygon, polyline, triangle, diamond, ellipse wider than 14 px);
     a page with fewer than --min-figures is reported (cards-only pages must be reworked).
  4. lively elements: per page, count how many kinds of lively effects appear (gradient, shadow/glow, star, polyline/polygon,
     dashed line, rotated sticker/stamp, mascot-sized ellipse group); pages with fewer than --min-lively kinds are reported;
     the deck must contain at least one rotated sticker/stamp.
  5. reveal (progressive appearance): every initiallyHidden node is revealed by some step; every revealed node exists, is
     initiallyHidden and belongs to the step's scene; no node is revealed twice; per page at least --min-beats reveal steps;
     every reveal step has speaker notes.
Usage:  python3 check_deck_quality.py deck.jbd.json [--min-figures 4] [--page-width 1280]
Exit code 1 if check 1, 2 or 5 (reveal structure) fails; checks 3 and 4 only warn (exit 0) unless --strict.
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
    ap.add_argument("--page-width", type=int, default=1280); ap.add_argument("--min-lively", type=int, default=3); ap.add_argument("--min-beats", type=int, default=2); ap.add_argument("--strict", action="store_true")
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
    # ---- 4. lively elements
    def kinds(ns):
        k = set()
        for n in ns:
            st = n.get("style", {}); sk = n.get("shapeKind")
            if st.get("gradientTo"): k.add("gradient")
            if st.get("shadowColor"): k.add("shadow")
            if sk == "star": k.add("star")
            if sk in ("polyline", "polygon"): k.add("curve")
            if st.get("borderStyle") in ("dashed", "dotted"): k.add("dashed")
            if n["transform"].get("rotation"): k.add("rotated")
            if sk == "ellipse" and n["width"] > 14: k.add("ellipse")
        return k
    dull = [(key[1] + 1, len(kinds(ns))) for key, ns in sorted(pages.items()) if len(ns) >= 8 and len(kinds(ns)) < a.min_lively]
    print(("WARN  pages with fewer than %d kinds of lively effects (page, kinds): %s" % (a.min_lively, dull)) if dull else "ok    every page has varied lively effects")
    if not any(n["transform"].get("rotation") for n in d["nodes"].values() if n["sceneId"] != d["rootSceneId"]): print("WARN  no rotated sticker/stamp anywhere in the deck"); dull.append("no-sticker")
    if dull and a.strict: bad += 1
    # ---- 5. reveal
    hidden = {i for i, n in d["nodes"].items() if n.get("initiallyHidden")}; seen = collections.Counter(); rv_bad = []; beats = collections.Counter(); no_notes = 0
    for s_ in steps:
        r = s_.get("reveal")
        if not r: continue
        ids = [e["nodeId"] for e in r.get("enter", [])]
        for i in ids:
            seen[i] += 1
            if i not in d["nodes"]: rv_bad.append(("missing node", i))
            else:
                if d["nodes"][i]["sceneId"] != s_["target"]["sceneId"]: rv_bad.append(("other scene", i))
                if i not in hidden: rv_bad.append(("not initiallyHidden", i))
        for i in r.get("exitNodeIds", []):
            if i not in d["nodes"] or d["nodes"][i]["sceneId"] != s_["target"]["sceneId"]: rv_bad.append(("bad exit node", i))
        if ids:
            n0 = d["nodes"].get(ids[0]); 
            if n0: beats[(n0["sceneId"], int(n0["transform"]["x"] // a.page_width))] += 1
            if not (s_.get("speakerNotes") or "").strip(): no_notes += 1
    never = sorted(hidden - set(seen)); twice = [i for i, c in seen.items() if c > 1]
    if never: rv_bad.append(("hidden but never revealed", len(never)))
    if twice: rv_bad.append(("revealed more than once", len(twice)))
    if no_notes: rv_bad.append(("reveal steps without speaker notes", no_notes))
    if rv_bad: bad += 1; print("FAIL  reveal problems:", rv_bad[:8])
    else: print("ok    reveal structure is consistent (%d hidden nodes, all revealed once)" % len(hidden))
    few = [(key[1] + 1, beats.get(key, 0)) for key in sorted(pages) if len(pages[key]) >= 8 and beats.get(key, 0) < a.min_beats]
    print(("WARN  pages with fewer than %d reveal beats (page, beats): %s" % (a.min_beats, few)) if few else "ok    every page has enough reveal beats")
    if few and a.strict: bad += 1
    sys.exit(1 if bad else 0)
main()
